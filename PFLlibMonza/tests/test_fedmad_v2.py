"""Behavioral checks for legitimate heterogeneity, delayed poisoning and recovery."""

import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.madsystem.config import load_experiment_config
from flcore.madsystem.defenses import build_defenses
from flcore.madsystem.evaluation import ExperimentEvaluator, confusion_metrics
from flcore.madsystem.global_validator import GlobalValidator
from flcore.madsystem.memory import HistoryAwareMemory
from flcore.madsystem.meta_agent import RuleBasedMetaAgent
from flcore.madsystem.risk import HistoryAwareRiskEngine
from flcore.madsystem.sentinel import HistoryAwareSentinel
from flcore.clients.clientmaliciousavg import ClientMaliciousAVG
from flcore.servers.servermad import ServerMAD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/fedmad"))
from prepare_v2_datasets import prepare_dataset
from analyze_results import comparison_key, metric_statistics


def model(value):
    result = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        result.weight.fill_(value)
    return result


def step(memory, values, round_number, args=None):
    args = args or SimpleNamespace()
    rows = HistoryAwareSentinel().inspect(list(range(len(values))), [model(x) for x in values], model(0), memory)
    assessment = HistoryAwareRiskEngine(args).assess(rows, memory, round_number)
    for cid, row in rows.items():
        details = assessment["clients"][cid]
        memory.update(cid, row["features"], row["anomaly"], details["risk"], round_number,
                      details["suspicious_evidence"], details["state"])
    return rows, assessment


class V2Tests(unittest.TestCase):
    def test_stable_population_outlier_is_not_punished_as_poisoning(self):
        memory = HistoryAwareMemory()
        for round_number in range(20):
            rows, assessment = step(memory, [1, 1, 1, 1, 10], round_number)
        self.assertGreater(rows[4]["population_anomaly"], 0.5)
        self.assertEqual(rows[4]["signals"]["temporal"], 0)
        self.assertLess(assessment["clients"][4]["risk"], 0.3)
        self.assertEqual(memory.get(4).current_state, "NORMAL")
        self.assertEqual(memory.get(4).reputation, 1.0)

    def test_same_current_outlier_has_more_risk_when_individual_history_changes(self):
        stable = HistoryAwareMemory()
        changed = HistoryAwareMemory()
        for round_number in range(8):
            step(stable, [1, 1, 1, 1, 100], round_number)
            step(changed, [1, 1, 1, 1, 1], round_number)
        stable_rows, stable_risk = step(stable, [1, 1, 1, 1, 100], 8)
        changed_rows, changed_risk = step(changed, [1, 1, 1, 1, 100], 8)
        self.assertGreater(changed_rows[4]["signals"]["temporal"], stable_rows[4]["signals"]["temporal"] + 0.5)
        self.assertGreater(changed_risk["clients"][4]["risk"], stable_risk["clients"][4]["risk"] + 0.3)
        self.assertFalse(changed_risk["clients"][4]["confirmed"])
        self.assertEqual(changed.get(4).reputation, 1.0)
        for round_number in range(9, 20):
            _, assessment = step(changed, [1, 1, 1, 1, -100], round_number)
        self.assertEqual(assessment["clients"][4]["state"], "DEFENSE")
        self.assertLess(changed.get(4).reputation, 1.0)
        for round_number in range(20, 65):
            _, assessment = step(changed, [1, 1, 1, 1, 1], round_number)
        self.assertEqual(assessment["clients"][4]["state"], "NORMAL")

    def test_common_training_magnitude_drift_is_discounted(self):
        memory = HistoryAwareMemory()
        for round_number in range(5):
            step(memory, [1, 1, 1, 1, 1], round_number)
        rows, _ = step(memory, [2, 2, 2, 2, 2], 5)
        self.assertTrue(all(row["signals"]["temporal"] < 1e-6 for row in rows.values()))

    def test_window_uses_rounds_and_does_not_treat_absence_as_benign(self):
        memory = HistoryAwareMemory(window=5)
        memory.update(4, {"norm": 1, "cosine": 1, "distance": 0, "temporal": 0.8}, 0.9, 0.8, 1, True, "WATCH")
        recent, consecutive, count, persistence = memory.evidence(4, 8, True)
        self.assertEqual(count, 1)
        self.assertEqual(consecutive, 1)
        self.assertEqual(len(recent), 1)
        self.assertLess(persistence, 1)

    def test_nonfinite_update_is_evidence_but_first_event_is_not_confirmed(self):
        memory = HistoryAwareMemory()
        _, assessment = step(memory, [1, 1, float("nan")], 0)
        self.assertEqual(assessment["clients"][2]["risk"], 1)
        self.assertFalse(assessment["clients"][2]["confirmed"])

    def test_global_clipping_bounds_complete_update_not_each_parameter(self):
        base = torch.nn.Linear(1, 1)
        local = torch.nn.Linear(1, 1)
        with torch.no_grad():
            base.weight.zero_(); base.bias.zero_()
            local.weight.fill_(3); local.bias.fill_(4)
        candidate = build_defenses("v2")["clipping"].aggregate(base, [local], [1], [1], clip_norm=1)
        norm = torch.sqrt(candidate.weight.square().sum() + candidate.bias.square().sum())
        self.assertAlmostEqual(float(norm.detach()), 1.0, places=6)
        self.assertEqual(float(local.weight.detach()), 3.0)

    def test_evaluator_latency_censoring_and_benign_defense(self):
        evaluator = ExperimentEvaluator([2])
        evaluator.record_round(39, [2, 3], [False, False], {2: False, 3: False}, robust_defense=True)
        evaluator.record_round(40, [2], [False], {2: True}, {2: "scaling"})
        evaluator.record_round(41, [2], [False], {2: True}, {2: "scaling"})
        evaluator.record_round(42, [2], [True], {2: True}, {2: "scaling"}, True)
        evaluator.record_round(43, [2], [False], {2: True}, {2: "label"})
        summary = evaluator.summary()
        self.assertEqual(summary["latency_by_client_attack"][0]["latency"], 2)
        self.assertIsNone(summary["latency_by_client_attack"][1]["latency"])
        self.assertEqual(summary["latency_censored_fraction"], 0.5)
        self.assertEqual(summary["unnecessary_defense_rate"], 1.0)
        self.assertNotIn("latency_by_client_attack", evaluator.summary(False))
        self.assertEqual(confusion_metrics([False], [True])["fnr"], 1.0)

    def test_hierarchical_config_and_strict_unknown_options(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"mad_version": "v2", "clients": 20, "rounds": 40,
                                        "malicious_fraction": 0.2, "attack": {"start_round": 30},
                                        "sentinel": {"enabled": True, "history_alpha": 0.9}}))
            config = load_experiment_config(path)
            self.assertEqual(config["global_rounds"], 39)
            self.assertEqual(config["n_client_malicious"], 4)
            self.assertEqual(config["round_init_atk"], 29)
            path.write_text(json.dumps({"risk": {"unknown": 1}}))
            with self.assertRaises(ValueError):
                load_experiment_config(path)

    def test_v2_validator_does_not_fall_back_to_client_test_data(self):
        with self.assertRaises(ValueError):
            GlobalValidator([], SimpleNamespace(mad_version="v2"), "cpu")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation.npz"
            np.savez(path, x=np.ones((4, 1), dtype=np.float32), y=np.zeros(4, dtype=np.int64))
            validator = GlobalValidator([], SimpleNamespace(mad_version="v2", mad_validation_data=str(path)), "cpu")
            candidate = torch.nn.Linear(1, 2)
            with torch.no_grad():
                candidate.weight.fill_(float("nan"))
            accepted, reason, _ = validator.validate(candidate, torch.nn.Linear(1, 2))
            self.assertFalse(accepted)
            self.assertEqual(reason, "non_finite_parameters")

    def test_v2_meta_chooses_behavioral_pattern_without_attack_labels(self):
        policy = RuleBasedMetaAgent(SimpleNamespace(mad_version="v2"))
        selected = policy.select_defenses({"round_level": "MEDIUM", "fraction_anomalous": 0.1})
        self.assertEqual(selected[0], "multi_krum")
        selected = policy.select_defenses({"round_level": "MEDIUM", "fraction_anomalous": 0.5, "excessive_magnitude_fraction": 0.2})
        self.assertEqual(selected[0], "clipping_trimmed_mean")

    def test_population_only_exposes_stable_heterogeneity_and_population_ablation_removes_it(self):
        memory = HistoryAwareMemory()
        for round_number in range(10):
            step(memory, [1, 1, 1, 1, 100], round_number)
        locals_ = [model(x) for x in [1, 1, 1, 1, 100]]
        full = HistoryAwareSentinel().inspect(list(range(5)), locals_, model(0), memory)
        population_only = HistoryAwareSentinel().inspect(list(range(5)), locals_, model(0), memory, use_temporal=False)
        without_population = HistoryAwareSentinel(use_population=False).inspect(list(range(5)), locals_, model(0), memory)
        self.assertGreater(population_only[4]["anomaly"], full[4]["anomaly"] + 0.3)
        self.assertEqual(without_population[4]["anomaly"], 0)

    def test_external_validator_rejects_class_regression_with_tolerant_global_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation.npz"
            x = np.tile(np.eye(2, dtype=np.float32), (10, 1))
            np.savez(path, x=x, y=np.tile([0, 1], 10))
            args = SimpleNamespace(mad_version="v2", mad_validation_data=str(path), num_classes=2,
                                   mad_validation_loss_tolerance=100, mad_validation_accuracy_tolerance=1,
                                   mad_validation_class_min_examples=5, mad_validation_class_tolerance=0.2)
            validator = GlobalValidator([], args, "cpu")
            trusted = torch.nn.Linear(2, 2)
            with torch.no_grad():
                trusted.weight.copy_(torch.eye(2)); trusted.bias.zero_()
            candidate = copy.deepcopy(trusted)
            with torch.no_grad():
                candidate.bias[0] = 2
            accepted, reason, metrics = validator.validate(candidate, trusted)
            self.assertFalse(accepted)
            self.assertEqual(reason, "validation_class_regression:1")
            self.assertEqual(metrics["class_examples"]["1"], 10)

    def test_v2_low_risk_bypasses_meta_then_rejection_escalates_and_rolls_back(self):
        server = ServerMAD.__new__(ServerMAD)
        server.v2 = True
        server.global_model = model(0)
        server.trusted_model = copy.deepcopy(server.global_model)
        server.uploaded_models = [model(1)] * 3
        server.uploaded_ids, server.uploaded_weights = [0, 1, 2], [1 / 3] * 3
        server.meta_agent = RuleBasedMetaAgent(SimpleNamespace(mad_version="v2"))
        server.risk_engine = HistoryAwareRiskEngine(SimpleNamespace())
        server.defense_agents = build_defenses("v2")
        server.validator = SimpleNamespace(validate=lambda candidate, trusted: (True, "accepted", {}), commit=lambda metrics: None)
        server.ablate_meta = server.ablate_validator = server.ablate_history = server.ablate_reputation = False
        server.filter_medium_risk_updates = False
        server.byzantine_f, server.clip_norm, server.max_defense_attempts = 0, 1, 2
        assessment = {"round_level": "LOW", "max_risk": 0, "fraction_anomalous": 0,
                      "clients": {i: {"risk": 0, "reputation": 1, "level": "LOW"} for i in range(3)}}
        chosen, _, rolled_back, _ = server._aggregate_and_validate(assessment)
        self.assertEqual(chosen, "fedavg")
        self.assertFalse(rolled_back)
        self.assertEqual(server.meta_agent.activations, 0)
        server.validator.validate = lambda candidate, trusted: (False, "rejected", {})
        chosen, outcomes, rolled_back, _ = server._aggregate_and_validate(assessment)
        self.assertTrue(rolled_back)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(server.meta_agent.activations, 1)
        self.assertEqual(float(server.global_model.weight.item()), 1)

    def test_attack_schedule_is_private_reproducible_and_clears_upload_truth(self):
        client = ClientMaliciousAVG.__new__(ClientMaliciousAVG)
        client.model, client.latest_global_model = model(1), model(0)
        client.id, client.round_init_atk, client.rate_client_fake = 2, 3, 1.0
        client.mad_deterministic, client.mad_seed = True, 42
        client.atack, client.attack_noise_snr = "gaussian", 1.0
        first = client.send_local_model(4)
        torch.manual_seed(17)
        torch.rand(100)
        second = client.send_local_model(4)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertEqual(client.last_attack, "gaussian")
        self.assertEqual(float(client.model.weight.item()), 1)
        client.send_local_model(3)
        self.assertFalse(client.is_malicious)
        self.assertIsNone(client.last_attack)

    def test_prepared_data_is_disjoint_for_iid_and_dirichlet_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "test"):
                (root / "toy" / split).mkdir(parents=True)
                np.savez(root / "toy" / split / "0.npz", data={"x": np.arange(200, dtype=np.float32)[:, None], "y": np.tile([0, 1], 100)})
            original = (root / "toy/train/0.npz").read_bytes()
            for distribution in ("iid", 0.1):
                name, validation = prepare_dataset("toy", 3, distribution, 1, batch_size=2, dataset_root=root)
                with np.load(validation) as data:
                    seen = set(data["sample_ids"].tolist())
                for split in ("train", "test"):
                    for cid in range(3):
                        with np.load(root / name / split / f"{cid}.npz", allow_pickle=True) as data:
                            ids = set(data["sample_ids"].tolist())
                        self.assertFalse(ids & seen)
                        seen.update(ids)
                self.assertEqual(len(seen), 400)
                repeated, _ = prepare_dataset("toy", 3, distribution, 1, batch_size=2, dataset_root=root)
                self.assertEqual(name, repeated)
            self.assertEqual(original, (root / "toy/train/0.npz").read_bytes())

    def test_targeted_label_poisoning_trains_once_and_does_not_repeat_on_upload(self):
        client = ClientMaliciousAVG.__new__(ClientMaliciousAVG)
        client.model = model(1)
        client.id, client._mad_round, client.round_init_atk = 2, 4, 3
        client.v2, client.mad_deterministic, client.mad_seed = True, True, 1
        client.atack, client.rate_client_fake = "label", 1.0
        client.source_label, client.target_label = 0, 1
        client.learning_rate_decay = False
        client.train_time_cost = {'num_rounds': 0, 'total_cost': 0}
        calls = []
        client._train_with_targeted_label_flip = lambda source, target: calls.append((source, target))
        client.train()
        client.send_local_model(4)
        self.assertEqual(calls, [(0, 1)])
        self.assertEqual(client.train_time_cost['num_rounds'], 1)
        self.assertTrue(client.is_malicious)

    def test_five_seeds_are_grouped_without_mixing_protocols_or_missing_metrics(self):
        records = [{"dataset": f"prepared_seed{seed}", "source_dataset": "MNIST", "distribution": 0.1,
                    "seed": seed, "attack": "sign_flipping", "method": "full", "code_sha256": "code", "protocol_sha256": "protocol"}
                   for seed in range(1, 6)]
        self.assertEqual(len({comparison_key(record) for record in records}), 1)
        changed = dict(records[0], code_sha256="changed")
        self.assertNotEqual(comparison_key(changed), comparison_key(records[0]))
        stats = metric_statistics([{"accuracy": value / 10} for value in range(1, 6)], "accuracy")
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 0.3)
        self.assertGreater(stats["sd"], 0)
        self.assertEqual(metric_statistics([{"asr": None}] * 5, "asr")["n"], 0)


if __name__ == "__main__":
    unittest.main()
