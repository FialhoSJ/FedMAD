import copy
import json
import os
import time
from threading import Thread

import torch.nn as nn

from flcore.clients.clientmad import ClientMAD
from flcore.madsystem.defenses import build_defenses
from flcore.madsystem.global_validator import GlobalValidator
from flcore.madsystem.meta_agent import RuleBasedMetaAgent
from flcore.madsystem.monitoring import (
    GradientAgent,
    HistoryAgent,
    PerformanceAgent,
    RiskAssessment,
    SimilarityAgent,
    StatisticalAgent,
)
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import FedAvgCNN


class ServerMAD(Server):
    """Adaptive multi-agent defense loop for federated model updates."""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(ClientMAD)
        self.static_mode = args.algorithm == "MADStatic"
        self.fixed_defense = getattr(args, "mad_fixed_defense", "trimmed_mean")

        # Build the extra encoder only when the optional local SSL path is active.
        self.encoder = (
            self._build_encoder(args)
            if int(getattr(args, "ssl_epochs", 0)) > 0
            else None
        )
        self.history = []

        all_agents = {
            "gradient": GradientAgent(),
            "similarity": SimilarityAgent(),
            "statistical": StatisticalAgent(),
            "performance": PerformanceAgent(),
            "history": HistoryAgent(getattr(args, "bhv_lookback", 5)),
        }
        enabled = getattr(args, "mad_agents", "all")
        if isinstance(enabled, str):
            enabled = [name.strip().lower() for name in enabled.split(",")]
        else:
            enabled = [str(name).lower() for name in enabled]
        unknown_agents = set(enabled) - set(all_agents) - {"all"}
        if unknown_agents:
            raise ValueError(
                "Unknown FedMAD monitoring agent(s): "
                + ", ".join(sorted(unknown_agents))
            )
        self.agents = [
            agent for name, agent in all_agents.items()
            if "all" in enabled or name in enabled
        ]
        if self.static_mode:
            self.agents = []
        self.agent_names = [agent.name for agent in self.agents]
        self.risk_assessment = RiskAssessment(args)
        self.meta_agent = RuleBasedMetaAgent(args)
        self.defense_agents = build_defenses()
        self.validator = GlobalValidator(self.clients, args, self.device)
        self.trusted_model = copy.deepcopy(self.global_model)
        self.detection_log = []
        self.byzantine_f = max(0, int(getattr(args, "mad_byzantine_f", 1)))
        self.clip_norm = max(1e-8, float(getattr(args, "mad_clip_norm", 1.0)))

        print(f"[FedMAD] Monitoring agents: {', '.join(self.agent_names) or 'none'}")
        if self.static_mode:
            print(f"[FedMAD] Static defense baseline: {self.fixed_defense}")
        else:
            print("[FedMAD] Rule-based meta-agent, temporal risk memory, and validator ready.")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedMAD server and clients.")

    def _build_encoder(self, args):
        in_features = 3 if "Cifar10" in args.dataset else 1
        dim = 1600 if "Cifar10" in args.dataset else 1024
        encoder = FedAvgCNN(
            in_features=in_features, num_classes=args.num_classes, dim=dim
        )
        encoder.fc = nn.Identity()
        return encoder.to(args.device)

    def send_models(self):
        super().send_models()
        if self.encoder is not None:
            for client in self.clients:
                if hasattr(client, "set_encoder"):
                    client.set_encoder(self.encoder)

    def set_client_quarantine(self, client_id):
        if client_id not in self.client_quarantine_dict:
            return
        state = self.client_quarantine_dict[client_id]
        state["quarentena"] += 1
        state["roundsQuarent"] = state["quarentena"] * 2

    def decrease_quarentine(self, client_id):
        state = self.client_quarantine_dict.get(client_id)
        if state and state["roundsQuarent"] > 0:
            state["roundsQuarent"] -= 1

    def _performance_impacts(self):
        impacts = {}
        for client_id, client_model in zip(self.uploaded_ids, self.uploaded_models):
            if client_id not in self.validator.client_batches:
                self.validator.client_batches[client_id] = self.validator.capture_batches(
                    self.clients[client_id]
                )
            batches = self.validator.client_batches[client_id]
            impacts[client_id] = self.validator.loss_impact(
                client_model, self.global_model, batches
            )
        return impacts

    def _monitor_updates(self, round_number):
        metadata = {
            "round": round_number,
            "client_ids": list(self.uploaded_ids),
            "history": self.history,
            "quarantined_ids": [
                cid for cid, state in self.client_quarantine_dict.items()
                if state["roundsQuarent"] > 0
            ],
            "performance_impacts": self._performance_impacts(),
            "agent_names": self.agent_names,
        }
        client_scores = {cid: [] for cid in self.uploaded_ids}
        for agent in self.agents:
            scores = agent.analyze(
                self.uploaded_models, self.global_model, metadata
            )
            for cid, score in zip(self.uploaded_ids, scores):
                client_scores[cid].append(float(score))
        return metadata, client_scores, self.risk_assessment.assess(
            self.uploaded_ids, client_scores
        )

    def _adjusted_weights(self, assessment):
        weights = []
        for cid, weight in zip(self.uploaded_ids, self.uploaded_weights):
            risk = assessment["clients"].get(cid, {}).get("risk", 0.0)
            weights.append(float(weight) * max(0.1, 1.0 - 0.5 * risk))
        total = sum(weights)
        if total <= 0:
            return [1.0 / len(weights)] * len(weights) if weights else []
        return [weight / total for weight in weights]

    @staticmethod
    def _model_communication_bytes(model):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.state_dict().values()
        )

    def _effective_defense_name(self, requested):
        count = len(self.uploaded_models)
        if requested in ("krum", "multi_krum") and count < 3:
            return "fedavg"
        if requested == "bulyan" and (
            self.byzantine_f == 0 or count < 4 * self.byzantine_f + 3
        ):
            return "trimmed_mean"
        return requested

    def _aggregate_and_validate(self, assessment):
        attempted = []
        outcomes = []
        candidates = self.meta_agent.select_defenses(assessment["round_level"])
        effective_weights = self._adjusted_weights(assessment)
        while candidates:
            defense_name = candidates.pop(0)
            if defense_name in attempted:
                continue
            attempted.append(defense_name)
            defense = self.defense_agents[defense_name]
            risk_scores = {
                cid: assessment["clients"].get(cid, {}).get("risk", 0.0)
                for cid in self.uploaded_ids
            }
            candidate = defense.aggregate(
                server_model=self.global_model,
                client_models=self.uploaded_models,
                weights=effective_weights,
                client_ids=self.uploaded_ids,
                risk_scores=risk_scores,
                byzantine_f=self.byzantine_f,
                clip_norm=self.clip_norm,
            )
            applied_name = self._effective_defense_name(defense_name)
            accepted, reason, metrics = self.validator.validate(
                candidate, self.trusted_model
            )
            outcomes.append({
                "defense": defense_name,
                "applied_defense": applied_name,
                "accepted": bool(accepted),
                "reason": reason,
                "validation": metrics,
            })
            print(
                f"[FedMAD] {assessment['round_level']} risk: "
                f"{defense_name} -> {'accepted' if accepted else reason}"
            )
            if accepted:
                round_reference = self.global_model
                self.global_model = candidate
                self.trusted_model = copy.deepcopy(candidate)
                self.validator.commit(metrics)
                self.defense_agents["foolsgold"].observe(
                    self.uploaded_models, round_reference, self.uploaded_ids
                )
                return applied_name, outcomes, False
            candidates.extend(
                self.meta_agent.fallback_defenses(
                    assessment["round_level"], attempted + candidates
                )
            )

        self.global_model = copy.deepcopy(self.trusted_model)
        return None, outcomes, True

    def _write_detection_log(self):
        result_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results"
        ))
        os.makedirs(result_dir, exist_ok=True)
        log_filename = (
            f"detection_log_{self.dataset}_{self.algorithm}_cc{self.cc}.json"
        )
        log_path = os.path.join(result_dir, log_filename)
        with open(log_path, "w", encoding="utf-8") as output:
            json.dump(self.detection_log, output, indent=2)
        print(f"Detection log saved -> {log_path}")

    def train(self):
        for round_number in range(self.global_rounds + 1):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if round_number % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_number}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client_id in range(self.num_clients):
                self.decrease_quarentine(client_id)

            threads = [Thread(target=client.train) for client in self.selected_clients]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            if self.selected_clients:
                self.receive_models()
            else:
                self.uploaded_ids = []
                self.uploaded_weights = []
                self.uploaded_models = []
                self.ids = []
            round_log = {
                "round": round_number,
                "agent_names": self.agent_names,
                "client_ids": list(self.uploaded_ids),
                "malicious_ground_truth": [
                    int(cid in self.index_malicious) for cid in self.uploaded_ids
                ],
                "per_agent_scores": {},
                "final_scores": {},
                "client_risks": {},
                "client_levels": {},
                "client_history": {},
                "round_risk": 0.0,
                "round_level": "LOW",
                "defense_attempts": [],
                "chosen_defense": None,
                "rollback": False,
                "monitoring_seconds": 0.0,
                "aggregation_validation_seconds": 0.0,
                "communication_bytes_download": (
                    self._model_communication_bytes(self.global_model)
                    + (
                        self._model_communication_bytes(self.encoder)
                        if self.encoder is not None else 0
                    )
                ) * len(self.clients),
                "communication_bytes_upload": sum(
                    self._model_communication_bytes(model)
                    for model in self.uploaded_models
                ),
                "quarantine_before": {
                    str(cid): self.client_quarantine_dict[cid]["roundsQuarent"]
                    for cid in range(self.num_clients)
                },
                "quarantined_clients": [],
            }

            if self.uploaded_models and self.static_mode:
                aggregation_start = time.time()
                round_reference = self.global_model
                defense = self.defense_agents[self.fixed_defense]
                candidate = defense.aggregate(
                    server_model=self.global_model,
                    client_models=self.uploaded_models,
                    weights=self.uploaded_weights,
                    client_ids=self.uploaded_ids,
                    byzantine_f=self.byzantine_f,
                    clip_norm=self.clip_norm,
                )
                applied_name = self._effective_defense_name(self.fixed_defense)
                self.global_model = candidate
                self.trusted_model = copy.deepcopy(candidate)
                if self.fixed_defense == "foolsgold":
                    defense.observe(self.uploaded_models, round_reference, self.uploaded_ids)
                round_log["round_level"] = "STATIC"
                round_log["chosen_defense"] = applied_name
                round_log["defense_attempts"] = [{
                    "defense": self.fixed_defense,
                    "applied_defense": applied_name,
                    "accepted": True,
                    "reason": "static_baseline",
                    "validation": None,
                }]
                round_log["aggregation_validation_seconds"] = (
                    time.time() - aggregation_start
                )
            elif self.uploaded_models:
                monitoring_start = time.time()
                _, client_scores, assessment = self._monitor_updates(round_number)
                round_log["monitoring_seconds"] = time.time() - monitoring_start
                round_log["final_scores"] = {
                    str(cid): float(assessment["raw_scores"].get(cid, 0.0))
                    for cid in self.uploaded_ids
                }
                round_log["client_risks"] = {
                    str(cid): float(details["risk"])
                    for cid, details in assessment["clients"].items()
                }
                round_log["client_levels"] = {
                    str(cid): details["level"]
                    for cid, details in assessment["clients"].items()
                }
                round_log["client_history"] = {
                    str(cid): {
                        "observations": int(details["observations"]),
                        "alerts": int(details["alerts"]),
                        "high_streak": int(details["high_streak"]),
                    }
                    for cid, details in assessment["clients"].items()
                }
                round_log["round_risk"] = float(assessment["round_risk"])
                round_log["round_level"] = assessment["round_level"]
                for agent_index, agent_name in enumerate(self.agent_names):
                    round_log["per_agent_scores"][agent_name] = {
                        str(cid): float(client_scores[cid][agent_index])
                        for cid in self.uploaded_ids
                    }

                aggregation_start = time.time()
                chosen, outcomes, rolled_back = self._aggregate_and_validate(assessment)
                round_log["aggregation_validation_seconds"] = (
                    time.time() - aggregation_start
                )
                round_log["chosen_defense"] = chosen
                round_log["defense_attempts"] = outcomes
                round_log["rollback"] = rolled_back

                # Quarantine only after persistent HIGH risk; a single unusual
                # Non-IID update changes the defense level without banning a client.
                for cid, details in assessment["clients"].items():
                    if (
                        details["confirmed"]
                        and self.client_quarantine_dict[cid]["roundsQuarent"] == 0
                    ):
                        self.set_client_quarantine(cid)
                        round_log["quarantined_clients"].append(int(cid))
            else:
                print("[FedMAD] No client updates received; keeping trusted model.")

            round_log["quarantine_after"] = {
                str(cid): self.client_quarantine_dict[cid]["roundsQuarent"]
                for cid in range(self.num_clients)
            }
            round_log["removed_clients"] = []
            self.detection_log.append(round_log)

            self.Budget.append(time.time() - start_time)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy:", max(self.rs_test_acc))
        average_time = sum(self.Budget[1:]) / max(1, len(self.Budget[1:]))
        print("Average time cost per round:", average_time)
        self.save_results()
        self.save_global_model()
        self._write_detection_log()
