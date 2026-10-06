import copy
import csv
import json
import os
import re
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import torch

import torch.nn as nn

from flcore.clients.clientmad import ClientMAD
from flcore.madsystem.defenses import build_defenses
from flcore.madsystem.global_validator import GlobalValidator
from flcore.madsystem.memory import MemoryManager, HistoryAwareMemory
from flcore.madsystem.meta_agent import RuleBasedMetaAgent
from flcore.madsystem.risk import RiskEngine, HistoryAwareRiskEngine
from flcore.madsystem.sentinel import FedMADSentinel, HistoryAwareSentinel
from flcore.madsystem.evaluation import ExperimentEvaluator
from flcore.madsystem.resources import process_memory
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import FedAvgCNN


class ServerMAD(Server):
    """Adaptive multi-agent defense loop for federated model updates."""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(ClientMAD)
        self.static_mode = args.algorithm == "MADStatic"
        self.v2 = getattr(args, "mad_version", "v1") == "v2"
        self.fixed_defense = getattr(args, "mad_fixed_defense", "trimmed_mean")
        self.run_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(getattr(args, "mad_run_id", "")))[:80]
        self.ablate_history = bool(getattr(args, "mad_ablate_history", False))
        self.ablate_reputation = bool(getattr(args, "mad_ablate_reputation", False))
        self.ablate_temporal = bool(getattr(args, "mad_ablate_temporal", False))
        self.ablate_meta = bool(getattr(args, "mad_ablate_meta", False))
        self.ablate_validator = bool(getattr(args, "mad_ablate_validator", False))
        self.ablate_population = bool(getattr(args, "mad_ablate_population", False))
        if self.v2 and not getattr(args, "mad_validator_enabled", True):
            self.ablate_validator = True
        self.evaluator = ExperimentEvaluator(self.index_malicious)

        # Build the extra encoder only when the optional local SSL path is active.
        self.encoder = (
            self._build_encoder(args)
            if int(getattr(args, "ssl_epochs", 0)) > 0
            else None
        )
        self.history = []

        self.sentinel = FedMADSentinel()
        self.memory = MemoryManager(
            alpha=getattr(args, "mad_history_alpha", 0.9),
            penalty=getattr(args, "mad_reputation_penalty", 0.08),
            recovery=getattr(args, "mad_reputation_recovery", 0.02),
            suspicious_threshold=getattr(args, "mad_anomaly_threshold", 0.45),
        )
        self.risk_engine = RiskEngine(
            low_threshold=getattr(args, "mad_low_threshold", 0.35),
            high_threshold=getattr(args, "mad_high_threshold", 0.65),
            client_threshold=getattr(args, "mad_client_threshold", 0.35),
            anomaly_threshold=getattr(args, "mad_anomaly_threshold", 0.45),
            high_patience=getattr(args, "mad_high_patience", 2),
            use_reputation=not (self.ablate_history or self.ablate_reputation),
            use_temporal=not (self.ablate_history or self.ablate_temporal),
        )
        if self.v2:
            self.sentinel = HistoryAwareSentinel(
                getattr(args, "mad_anomaly_weights", None), getattr(args, "mad_population_floor", 0.15),
                getattr(args, "mad_cold_start_factor", 0.5), getattr(args, "mad_history_min_observations", 3),
                getattr(args, "mad_history_sensitivity", 3.0), not self.ablate_population,
            )
            self.memory = HistoryAwareMemory(
                getattr(args, "mad_history_alpha", 0.9), getattr(args, "mad_reputation_penalty", 0.08),
                getattr(args, "mad_reputation_recovery", 0.02), getattr(args, "mad_anomaly_threshold", 0.45),
                getattr(args, "mad_persistence_window", 5), getattr(args, "mad_reputation_initial", 1.0),
                getattr(args, "mad_suspicious_profile_weight", 0.1),
            )
            self.risk_engine = HistoryAwareRiskEngine(args, not (self.ablate_history or self.ablate_reputation), not (self.ablate_history or self.ablate_temporal))
        self.agent_names = [] if self.static_mode else ["FedMAD Sentinel"]
        if self.v2 and not self.static_mode:
            self.agent_names.append("FedMAD Meta-Defense")
        self.feature_names = self.sentinel.feature_names
        self.meta_agent = RuleBasedMetaAgent(args)
        self.defense_agents = build_defenses("v2" if self.v2 else "v1")
        self.defense_pool = self.defense_agents  # Compatibility alias; strategies are not agents.
        self.validator = GlobalValidator(self.clients, args, self.device)
        self.trusted_model = copy.deepcopy(self.global_model)
        self.detection_log = []
        self.byzantine_f = max(0, int(getattr(args, "mad_byzantine_f", 1)))
        self.last_accepted_defense = None
        self.filter_medium_risk_updates = bool(
            getattr(args, "mad_filter_medium_risk_updates", True)
        )
        self.clip_norm = max(1e-8, float(getattr(args, "mad_clip_norm", 1.0)))
        self.max_defense_attempts = max(1, int(getattr(args, "mad_max_defense_attempts", 4)))
        self.quarantine_rounds = max(0, int(getattr(args, "mad_quarantine_rounds", 0)))
        self.evaluation_every = max(1, int(getattr(args, "mad_evaluation_every", 1)))
        self.last_monitoring_timings = {}
        if self.v2:
            self._initialize_v2_logs()

        print(f"[FedMAD] Monitoring agent: {', '.join(self.agent_names) or 'none (static)'}")
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
        if self.v2:
            state["roundsQuarent"] = self.quarantine_rounds
            return
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
        prior_memory = self.memory if not self.ablate_history else (HistoryAwareMemory() if self.v2 else MemoryManager(
            alpha=self.memory.alpha, penalty=self.memory.penalty,
            recovery=self.memory.recovery,
            suspicious_threshold=self.memory.suspicious_threshold,
        ))
        sentinel_start = time.perf_counter()
        observations = self.sentinel.inspect(
            self.uploaded_ids, self.uploaded_models, self.global_model, prior_memory,
            use_temporal=not (self.ablate_history or self.ablate_temporal),
        )
        sentinel_seconds = time.perf_counter() - sentinel_start
        risk_start = time.perf_counter()
        assessment = self.risk_engine.assess(observations, self.memory, round_number) if self.v2 else self.risk_engine.assess(observations, prior_memory)
        for cid, row in observations.items():
            extra = {"suspicious": assessment["clients"][cid]["suspicious_evidence"],
                     "current_state": assessment["clients"][cid]["state"],
                     "use_reputation": not (self.ablate_history or self.ablate_reputation)} if self.v2 else {}
            state = self.memory.update(
                cid, row["features"], row["anomaly"],
                assessment["clients"][cid]["risk"], round_number, **extra,
            )
            assessment["clients"][cid].update({
                "reputation": state.reputation,
                "observations": state.observations,
                "alerts": state.total_suspicious_rounds,
            })
        self.last_monitoring_timings = {"sentinel_seconds": sentinel_seconds, "memory_risk_seconds": time.perf_counter() - risk_start}
        return observations, assessment

    def _adjusted_weights(self, assessment):
        if assessment["round_level"] == "LOW" or (getattr(self, "v2", False) and self.ablate_meta):
            return list(self.uploaded_weights)
        weights = []
        for cid, weight in zip(self.uploaded_ids, self.uploaded_weights):
            risk = assessment["clients"].get(cid, {}).get("risk", 0.0)
            reputation = 1.0 if (self.ablate_history or self.ablate_reputation) else assessment["clients"].get(cid, {}).get("reputation", 1.0)
            weights.append(float(weight) * max(0.1, reputation) * max(0.1, 1.0 - 0.5 * risk))
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

    def _effective_defense_name(self, requested, byzantine_f=None):
        count = len(self.uploaded_models)
        byzantine_f = self.byzantine_f if byzantine_f is None else byzantine_f
        if requested in ("krum", "multi_krum") and count < 2 * byzantine_f + 3:
            return "trimmed_mean" if count > 1 else "fedavg"
        if requested == "foolsgold" and count < 2:
            return "fedavg"
        if requested in ("median", "trimmed_mean") and count == 1:
            return "fedavg"
        if requested == "bulyan" and (
            byzantine_f == 0 or count < 4 * byzantine_f + 3
        ):
            return "trimmed_mean"
        return requested

    def _risk_adjusted_byzantine_f(self, assessment):
        """Raise the configured f floor as active clients show stronger risk."""
        count = len(self.uploaded_models)
        if getattr(self, "v2", False) and self.ablate_meta:
            return self.byzantine_f
        if count <= 1:
            return 0
        levels = [
            assessment["clients"].get(cid, {}).get("level", "LOW")
            for cid in self.uploaded_ids
        ]
        high_count = sum(level == "HIGH" for level in levels)
        medium_count = sum(level == "MEDIUM" for level in levels)
        # Treat HIGH clients as one suspected Byzantine each and round up half
        # the MEDIUM count; configured f remains a minimum threat budget.
        risk_estimate = high_count + (medium_count + 1) // 2
        safe_trim_limit = (count - 1) // 2
        return min(max(self.byzantine_f, risk_estimate), safe_trim_limit)

    def _risk_filtered_indices(self, assessment):
        """Return a safe-enough LOW-risk subset for a validated candidate."""
        if (
            not self.filter_medium_risk_updates
            or assessment["round_level"] not in ("MEDIUM", "HIGH")
        ):
            return []
        indices = [
            index
            for index, client_id in enumerate(self.uploaded_ids)
            if assessment["clients"].get(client_id, {}).get("level", "LOW") == "LOW"
        ]
        minimum_retained = max(3, (len(self.uploaded_models) + 1) // 2)
        if len(indices) < minimum_retained or len(indices) == len(self.uploaded_models):
            return []
        return indices

    def _aggregate_and_validate(self, assessment):
        attempted = []
        outcomes = []
        selection_start = time.perf_counter()
        if self.ablate_meta:
            candidates = [self.fixed_defense]
            self.meta_agent.last_reason = "fixed_defense_ablation"
        elif getattr(self, "v2", False) and assessment["round_level"] == "LOW" and not assessment.get("fraction_anomalous", 0) and assessment.get("max_risk", 0) < self.risk_engine.client_threshold:
            candidates = ["fedavg"]
            self.meta_agent.last_reason = "low_effective_risk_meta_inactive"
        else:
            candidates = self.meta_agent.select_defenses(assessment)
        self.last_meta_seconds = time.perf_counter() - selection_start
        risk_filtered_indices = [] if self.ablate_meta else self._risk_filtered_indices(assessment)
        if risk_filtered_indices:
            candidates.insert(0, "risk_filtered_fedavg")
        effective_weights = self._adjusted_weights(assessment)
        effective_f = self._risk_adjusted_byzantine_f(assessment)
        while candidates and len(outcomes) < self.max_defense_attempts:
            defense_name = candidates.pop(0)
            if defense_name in attempted:
                continue
            is_risk_filtered = defense_name == "risk_filtered_fedavg"
            if is_risk_filtered:
                candidate_indices = risk_filtered_indices
                defense_key = "fedavg"
                applied_name = "fedavg"
                attempted.append(defense_name)
            else:
                candidate_indices = list(range(len(self.uploaded_models)))
                applied_name = self._effective_defense_name(defense_name, effective_f)
                defense_key = applied_name
                if applied_name in attempted:
                    attempted.append(defense_name)
                    continue
                attempted.append(defense_name)
                if applied_name != defense_name:
                    # For example, Bulyan falls back to trimmed mean when the
                    # participant count cannot support its Byzantine bound.
                    attempted.append(applied_name)
            defense = self.defense_agents[defense_key]
            candidate_models = [self.uploaded_models[index] for index in candidate_indices]
            candidate_ids = [self.uploaded_ids[index] for index in candidate_indices]
            candidate_weights = [effective_weights[index] for index in candidate_indices]
            risk_scores = {
                client_id: assessment["clients"].get(client_id, {}).get("risk", 0.0)
                for client_id in candidate_ids
            }
            aggregation_start = time.perf_counter()
            candidate = defense.aggregate(
                server_model=self.global_model,
                client_models=candidate_models,
                weights=candidate_weights,
                client_ids=candidate_ids,
                risk_scores=risk_scores,
                byzantine_f=effective_f,
                clip_norm=self.clip_norm,
            )
            aggregation_seconds = time.perf_counter() - aggregation_start
            validation_start = time.perf_counter()
            if self.ablate_validator:
                accepted, reason, metrics = True, "validator_disabled_ablation", {}
            else:
                accepted, reason, metrics = self.validator.validate(
                    candidate, self.trusted_model
                )
            outcomes.append({
                "defense": defense_name,
                "applied_defense": applied_name,
                "aggregation_scope": "low_risk_subset" if is_risk_filtered else "all_active_clients",
                "excluded_client_ids": sorted(set(self.uploaded_ids) - set(candidate_ids)),
                "byzantine_f": effective_f,
                "accepted": bool(accepted),
                "reason": reason,
                "validation": metrics,
                "aggregation_seconds": aggregation_seconds,
                "validation_seconds": time.perf_counter() - validation_start,
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
                self.last_accepted_defense = applied_name
                chosen_name = defense_name if is_risk_filtered else applied_name
                return chosen_name, outcomes, False, effective_f
            if not self.ablate_meta:
                fallback_start = time.perf_counter()
                if getattr(self, "v2", False) and self.meta_agent.last_reason == "low_effective_risk_meta_inactive":
                    candidates.extend(self.meta_agent.select_defenses(assessment))
                else:
                    candidates.extend(
                    self.meta_agent.fallback_defenses(assessment["round_level"], attempted + candidates)
                    )
                self.last_meta_seconds += time.perf_counter() - fallback_start

        self.global_model = copy.deepcopy(self.trusted_model)
        return None, outcomes, True, effective_f

    def _write_detection_log(self):
        result_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results"
        ))
        os.makedirs(result_dir, exist_ok=True)
        log_filename = (
            f"detection_log_{self.dataset}_{self.algorithm}_cc{self.cc}"
            f"{'_' + self.run_id if self.run_id else ''}.json"
        )
        log_path = os.path.join(result_dir, log_filename)
        with open(log_path, "w", encoding="utf-8") as output:
            json.dump(self.detection_log, output, indent=2)
        print(f"Detection log saved -> {log_path}")
        if self.v2:
            summary = self.evaluator.summary()
            totals = {key: sum(item.get("detection", {}).get(key, 0) for item in self.detection_log) for key in ("tp", "fp", "tn", "fn")}
            summary.update(totals)
            summary["peak_memory_bytes"] = process_memory()["peak_memory_bytes"]
            summary["sentinel_seconds"] = sum(row.get("sentinel_seconds", 0) for row in self.detection_log)
            summary["monitoring_seconds"] = sum(row.get("monitoring_seconds", 0) for row in self.detection_log)
            summary["round_seconds"] = sum(self.Budget)
            with open(log_path.replace("detection_log_", "evaluation_") , "w", encoding="utf-8") as output:
                json.dump(summary, output, indent=2)

    def _initialize_v2_logs(self):
        result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))
        os.makedirs(result_dir, exist_ok=True)
        prefix = f"detection_log_{self.dataset}_{self.algorithm}_cc{self.cc}{'_' + self.run_id if self.run_id else ''}"
        self.round_jsonl = os.path.join(result_dir, prefix + ".jsonl")
        self.client_csv = os.path.join(result_dir, prefix + "_clients.csv")
        self.client_log_columns = ["round", "client_id", "population_deviation", "historical_deviation", "anomaly", "population_anomaly", "population_gate", "reputation", "risk", "state", "persistence", "suspicious_in_window", "round_risk", "chosen_defense", "validator"]
        with open(self.round_jsonl, "w", encoding="utf-8"):
            pass
        with open(self.client_csv, "w", newline="", encoding="utf-8") as output:
            csv.DictWriter(output, self.client_log_columns).writeheader()

    def _append_v2_logs(self, round_log):
        with open(self.round_jsonl, "a", encoding="utf-8") as output:
            output.write(json.dumps(round_log) + "\n")
        with open(self.client_csv, "a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, self.client_log_columns)
            for cid in round_log["client_ids"]:
                key = str(cid)
                features = round_log["sentinel_features"].get(key, {})
                history = round_log["client_history"].get(key, {})
                evidence = round_log.get("sentinel_evidence", {}).get(key, {})
                writer.writerow({"round": round_log["round"], "client_id": cid,
                                 "population_deviation": features.get("distance"), "historical_deviation": features.get("temporal"),
                                 "anomaly": round_log["final_scores"].get(key), "population_anomaly": evidence.get("population_anomaly"),
                                 "population_gate": evidence.get("population_gate"), "reputation": round_log["client_reputations"].get(key),
                                 "risk": round_log["client_risks"].get(key), "state": round_log.get("client_states", {}).get(key),
                                 "persistence": history.get("persistence"), "suspicious_in_window": history.get("suspicious_in_window"),
                                 "round_risk": round_log["round_risk"], "chosen_defense": round_log["chosen_defense"],
                                 "validator": "ROLLBACK" if round_log["rollback"] else "ACCEPT"})

    def train(self):
        for round_number in range(self.global_rounds + 1):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if round_number % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_number}-------------")
                print("\nEvaluate global model")
                if not self.v2:
                    self.evaluate()

            for client_id in range(self.num_clients):
                self.decrease_quarentine(client_id)

            if self.v2 and getattr(self.args, "mad_deterministic", False):
                for client in self.selected_clients:
                    torch.manual_seed(self.args.seed * 1000003 + round_number * 100003 + client.id * 1009)
                    client.train()
            elif self.v2:
                with ThreadPoolExecutor(max_workers=max(1, len(self.selected_clients))) as pool:
                    list(pool.map(lambda client: client.train(), self.selected_clients))
            else:
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
                "ablations": {
                    "history": self.ablate_history,
                    "reputation": self.ablate_reputation,
                    "temporal": self.ablate_temporal,
                    "population": self.ablate_population,
                    "meta": self.ablate_meta,
                    "validator": self.ablate_validator,
                },
                "agent_names": self.agent_names,
                "client_ids": list(self.uploaded_ids),
                "malicious_ground_truth": [],
                "active_attack_ground_truth": [],
                "per_agent_scores": {},
                "final_scores": {},
                "client_risks": {},
                "client_levels": {},
                "client_history": {},
                "round_risk": 0.0,
                "round_level": "LOW",
                "round_risk_components": {},
                "client_reputations": {},
                "sentinel_features": {},
                "effective_byzantine_f": None,
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
                applied_name = self._effective_defense_name(self.fixed_defense)
                defense = self.defense_agents[applied_name]
                candidate = defense.aggregate(
                    server_model=self.global_model,
                    client_models=self.uploaded_models,
                    weights=self.uploaded_weights,
                    client_ids=self.uploaded_ids,
                    byzantine_f=self.byzantine_f,
                    clip_norm=self.clip_norm,
                )
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
                round_log["meta_seconds"] = 0.0
            elif self.uploaded_models:
                monitoring_start = time.time()
                observations, assessment = self._monitor_updates(round_number)
                round_log["monitoring_seconds"] = time.time() - monitoring_start
                round_log.update(self.last_monitoring_timings)
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
                round_log["round_risk_components"] = {
                    key: float(assessment[key])
                    for key in ("mean_risk", "max_risk", "fraction_high", "max_anomaly", "fraction_anomalous")
                }
                round_log["client_reputations"] = {
                    str(cid): float(details["reputation"])
                    for cid, details in assessment["clients"].items()
                }
                round_log["sentinel_features"] = {
                    str(cid): {name: value for name, value in observations[cid]["features"].items() if not name.startswith("_")} for cid in self.uploaded_ids
                }
                round_log["per_agent_scores"]["FedMAD Sentinel"] = {
                    str(cid): observations[cid]["signals"] for cid in self.uploaded_ids
                }
                if self.v2:
                    round_log["client_states"] = {str(cid): details["state"] for cid, details in assessment["clients"].items()}
                    round_log["client_risk_components"] = {str(cid): details["risk_components"] for cid, details in assessment["clients"].items()}
                    round_log["sentinel_evidence"] = {str(cid): {key: observations[cid][key] for key in ("population_anomaly", "population_gate", "history_mature", "invalid_update")} for cid in self.uploaded_ids}
                    for cid, details in assessment["clients"].items():
                        round_log["client_history"][str(cid)].update({key: details[key] for key in ("persistence", "suspicious_in_window")})
                    round_log["round_risk_components"].update({key: assessment[key] for key in ("risk_variance", "high_risk_count")})

                aggregation_start = time.time()
                chosen, outcomes, rolled_back, effective_f = self._aggregate_and_validate(assessment)
                round_log["aggregation_validation_seconds"] = (
                    time.time() - aggregation_start
                )
                round_log["chosen_defense"] = chosen
                round_log["meta_seconds"] = self.last_meta_seconds
                round_log["meta_decision_reason"] = self.meta_agent.last_reason
                round_log["effective_byzantine_f"] = effective_f
                round_log["defense_attempts"] = outcomes
                round_log["rollback"] = rolled_back

                # Quarantine only after persistent HIGH risk; a single unusual
                # Non-IID update changes the defense level without banning a client.
                for cid, details in assessment["clients"].items():
                    if (
                        details["confirmed"]
                        and (not self.v2 or self.quarantine_rounds > 0)
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
            round_log["robust_defense_activated"] = bool(
                round_log["chosen_defense"]
                and round_log["chosen_defense"] not in ("fedavg", "risk_filtered_fedavg")
            )
            if self.v2:
                round_log["planned_client_ids"] = getattr(self, "planned_client_ids", [client.id for client in self.selected_clients])
                round_log["robust_defense_activated"] = any(outcome["defense"] != "fedavg" for outcome in round_log["defense_attempts"])
                round_log["mad_version"] = "v2"
                round_log["meta_activations"] = self.meta_agent.activations
                if round_number % self.evaluation_every == 0 or round_number == self.global_rounds:
                    self.evaluate()
                    round_log["global_metrics"] = self.latest_global_metrics
            predicted = [
                int(round_log.get("client_states", {}).get(str(cid)) == "DEFENSE") if self.v2 else
                int(round_log["client_risks"].get(str(cid), 0.0) >= self.risk_engine.client_threshold)
                for cid in self.uploaded_ids
            ]
            risk_predictions = [int(round_log["client_risks"].get(str(cid), 0) >= self.risk_engine.client_threshold) for cid in self.uploaded_ids] if self.v2 and not self.static_mode else None
            round_log.update(self.evaluator.record_server_round(round_number, self.uploaded_ids, predicted, self.clients, round_log["robust_defense_activated"], risk_predictions))
            eligible_rounds = sum(bool(item["client_ids"]) for item in self.detection_log) + bool(round_log["client_ids"])
            activated_rounds = sum(bool(item.get("robust_defense_activated")) for item in self.detection_log) + round_log["robust_defense_activated"]
            round_log["defense_activation_rate_so_far"] = (
                activated_rounds / eligible_rounds if eligible_rounds else 0.0
            )
            self.detection_log.append(round_log)

            self.Budget.append(time.time() - start_time)
            round_log["round_seconds"] = self.Budget[-1]
            if self.v2:
                round_log.update(process_memory())
                round_log["sentinel_overhead"] = round_log.get("sentinel_seconds", 0) / max(round_log["round_seconds"], 1e-12)
                self._append_v2_logs(round_log)
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
