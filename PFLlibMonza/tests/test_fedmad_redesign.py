"""Small deterministic checks for the two-agent FedMAD round core."""

import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))

from flcore.madsystem.defenses import build_defenses
from flcore.madsystem.memory import MemoryManager
from flcore.madsystem.meta_agent import RuleBasedMetaAgent
from flcore.madsystem.risk import RiskEngine
from flcore.madsystem.sentinel import FedMADSentinel
from flcore.attack.attack import scaled_update_model
from flcore.servers.servermad import ServerMAD


def model(value):
    result = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        result.weight.fill_(value)
    return result


class RedesignTests(unittest.TestCase):
    def test_sign_flip_and_scaling_transform_updates_without_mutation(self):
        base = model(2)
        local = model(3)
        self.assertAlmostEqual(float(scaled_update_model(local, base, -1).weight.item()), 1)
        self.assertAlmostEqual(float(scaled_update_model(local, base, 10).weight.item()), 12)
        self.assertAlmostEqual(float(local.weight.item()), 3)

    def test_sentinel_scores_outlier_and_uses_previous_history(self):
        sentinel = FedMADSentinel()
        memory = MemoryManager()
        ids = list(range(5))
        base = model(0)
        normal = [model(1) for _ in ids]
        first = sentinel.inspect(ids, normal, base, memory)
        self.assertTrue(all(row["anomaly"] == 0 for row in first.values()))
        self.assertTrue(all(row["signals"]["temporal"] == 0 for row in first.values()))
        for cid, row in first.items():
            memory.update(cid, row["features"], row["anomaly"], 0, 0)
        changed = sentinel.inspect(ids, normal[:4] + [model(10)], base, memory)
        self.assertGreater(changed[4]["anomaly"], 0.45)
        self.assertGreater(changed[4]["signals"]["temporal"], 0)
        self.assertTrue(all(changed[cid]["anomaly"] < changed[4]["anomaly"] for cid in ids[:4]))
        no_temporal = sentinel.inspect(ids, normal[:4] + [model(10)], base, memory, use_temporal=False)
        self.assertEqual(no_temporal[4]["signals"]["temporal"], 0)

    def test_memory_reputation_penalty_recovery_and_ema(self):
        memory = MemoryManager(alpha=0.5, penalty=0.2, recovery=0.05, suspicious_threshold=0.5)
        state = memory.update(7, {"norm": 2, "cosine": 1, "distance": 0}, 1, 0.8, 0)
        self.assertAlmostEqual(state.reputation, 0.8)
        self.assertEqual(state.consecutive_suspicious_rounds, 1)
        state = memory.update(7, {"norm": 4, "cosine": 1, "distance": 0}, 0, 0.1, 1)
        self.assertAlmostEqual(state.norm_ema, 3)
        self.assertAlmostEqual(state.reputation, 0.85)
        self.assertEqual(state.consecutive_suspicious_rounds, 0)
        self.assertEqual(state.total_suspicious_rounds, 1)

    def test_risk_distinguishes_isolated_and_population_anomalies(self):
        engine = RiskEngine()
        memory = MemoryManager()
        def row(score):
            return {"anomaly": score, "signals": {"temporal": 0}}
        isolated = engine.assess({i: row(0.8 if i == 4 else 0) for i in range(5)}, memory)
        broad = engine.assess({i: row(0.8) for i in range(5)}, memory)
        self.assertLess(isolated["round_risk"], broad["round_risk"])
        self.assertEqual(isolated["fraction_anomalous"], 0.2)
        self.assertEqual(broad["fraction_anomalous"], 1.0)
        memory.update(0, {"norm": 1, "cosine": 1, "distance": 0}, 1, 0.5, 0)
        without_reputation = RiskEngine(use_reputation=False).assess({0: row(0)}, memory)
        self.assertEqual(without_reputation["clients"][0]["risk"], 0)

    def test_meta_policy_uses_distribution_and_normal_round(self):
        policy = RuleBasedMetaAgent(SimpleNamespace())
        self.assertEqual(policy.select_defenses({"round_level": "LOW"})[0], "fedavg")
        self.assertEqual(policy.select_defenses({
            "round_level": "LOW", "max_anomaly": 0.8, "fraction_anomalous": 0.2,
        })[0], "multi_krum")
        self.assertEqual(policy.select_defenses({
            "round_level": "MEDIUM", "max_anomaly": 0.8, "fraction_anomalous": 0.8,
        })[0], "trimmed_mean")

    def test_rejected_candidates_restore_trusted_model(self):
        server = ServerMAD.__new__(ServerMAD)
        server.global_model = model(0)
        server.trusted_model = copy.deepcopy(server.global_model)
        server.uploaded_models = [model(1), model(1), model(1)]
        server.uploaded_ids = [0, 1, 2]
        server.uploaded_weights = [1 / 3] * 3
        server.meta_agent = RuleBasedMetaAgent(SimpleNamespace())
        server.defense_agents = build_defenses()
        server.validator = SimpleNamespace(validate=lambda candidate, trusted: (False, "rejected", {}))
        server.last_accepted_defense = None
        server.filter_medium_risk_updates = False
        server.byzantine_f = 0
        server.clip_norm = 1.0
        server.max_defense_attempts = 2
        server.ablate_meta = False
        server.ablate_validator = False
        assessment = {
            "round_level": "LOW", "clients": {i: {"risk": 0, "reputation": 1, "level": "LOW"} for i in range(3)},
        }
        chosen, outcomes, rolled_back, _ = server._aggregate_and_validate(assessment)
        self.assertIsNone(chosen)
        self.assertTrue(rolled_back)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(float(server.global_model.weight.item()), 0.0)

    def test_fixed_meta_and_validator_ablations_accept_one_candidate(self):
        server = ServerMAD.__new__(ServerMAD)
        server.global_model = model(0)
        server.trusted_model = copy.deepcopy(server.global_model)
        server.uploaded_models = [model(1), model(1), model(10)]
        server.uploaded_ids = [0, 1, 2]
        server.uploaded_weights = [1 / 3] * 3
        server.meta_agent = RuleBasedMetaAgent(SimpleNamespace())
        server.defense_agents = build_defenses()
        server.validator = SimpleNamespace(commit=lambda metrics: None)
        server.fixed_defense = "median"
        server.last_accepted_defense = None
        server.filter_medium_risk_updates = False
        server.byzantine_f = 0
        server.clip_norm = 1.0
        server.max_defense_attempts = 4
        server.ablate_meta = True
        server.ablate_validator = True
        server.ablate_history = False
        server.ablate_reputation = False
        assessment = {
            "round_level": "HIGH", "clients": {i: {"risk": 0.8, "reputation": 0.8, "level": "HIGH"} for i in range(3)},
        }
        chosen, outcomes, rolled_back, _ = server._aggregate_and_validate(assessment)
        self.assertEqual(chosen, "median")
        self.assertFalse(rolled_back)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["reason"], "validator_disabled_ablation")
        self.assertAlmostEqual(float(server.global_model.weight.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
