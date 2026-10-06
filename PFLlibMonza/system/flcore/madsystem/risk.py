"""Deterministic client and population risk estimates."""

from flcore.madsystem.memory import _clip
from flcore.madsystem.config import normalized_weights


class RiskEngine:
    def __init__(self, low_threshold=0.35, high_threshold=0.65, client_threshold=0.35, anomaly_threshold=0.45, high_patience=2, use_reputation=True, use_temporal=True):
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError("risk thresholds must satisfy 0 <= low <= high <= 1")
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.client_threshold = _clip(client_threshold)
        self.anomaly_threshold = _clip(anomaly_threshold)
        self.high_patience = max(1, int(high_patience))
        self.use_reputation = bool(use_reputation)
        self.use_temporal = bool(use_temporal)

    def level(self, value):
        if value >= self.high_threshold:
            return "HIGH"
        if value >= self.low_threshold:
            return "MEDIUM"
        return "LOW"

    def assess(self, observations, memory):
        clients = {}
        risks = []
        for cid, row in observations.items():
            prior = memory.get(cid)
            prior_rep = prior.reputation if prior is not None else 1.0
            anomaly = row["anomaly"]
            temporal = row["signals"]["temporal"]
            weights = (0.65, 0.20 if self.use_reputation else 0.0, 0.15 if self.use_temporal else 0.0)
            total_weight = sum(weights)
            risk = _clip((weights[0] * anomaly + weights[1] * (1.0 - prior_rep) + weights[2] * temporal) / total_weight)
            streak = (prior.consecutive_suspicious_rounds if prior is not None else 0) + 1 if anomaly >= memory.suspicious_threshold else 0
            clients[cid] = {
                "risk": risk,
                "level": self.level(risk),
                "reputation_before": prior_rep,
                "confirmed": streak >= self.high_patience,
                "high_streak": streak,
            }
            risks.append(risk)
        if risks:
            mean_risk = sum(risks) / len(risks)
            max_risk = max(risks)
            fraction_high = sum(value >= self.client_threshold for value in risks) / len(risks)
            max_anomaly = max(row["anomaly"] for row in observations.values())
            fraction_anomalous = sum(row["anomaly"] >= self.anomaly_threshold for row in observations.values()) / len(risks)
            round_risk = _clip(0.30 * mean_risk + 0.40 * max_risk + 0.30 * fraction_high)
        else:
            mean_risk = max_risk = fraction_high = max_anomaly = fraction_anomalous = round_risk = 0.0
        return {
            "clients": clients,
            "raw_scores": {cid: row["anomaly"] for cid, row in observations.items()},
            "round_risk": round_risk,
            "round_level": self.level(round_risk),
            "mean_risk": mean_risk,
            "max_risk": max_risk,
            "fraction_high": fraction_high,
            "max_anomaly": max_anomaly,
            "fraction_anomalous": fraction_anomalous,
        }


class HistoryAwareRiskEngine(RiskEngine):
    """Reversible states driven by joint evidence, never simulation labels."""

    def __init__(self, args, use_reputation=True, use_temporal=True):
        super().__init__(getattr(args, "mad_low_threshold", 0.35), getattr(args, "mad_high_threshold", 0.65),
                         getattr(args, "mad_suspicious_threshold", 0.60), getattr(args, "mad_anomaly_threshold", 0.45),
                         getattr(args, "mad_high_patience", 2), use_reputation, use_temporal)
        self.weights = normalized_weights(getattr(args, "mad_risk_weights", None), {"anomaly": 0.35, "history": 0.35, "reputation": 0.15, "persistence": 0.15})
        self.round_weights = normalized_weights(getattr(args, "mad_round_risk_weights", None), {"mean": 0.30, "max": 0.35, "fraction": 0.20, "variance": 0.15})
        self.watch_threshold = float(getattr(args, "mad_watch_threshold", 0.30))
        self.suspicious_threshold = float(getattr(args, "mad_suspicious_threshold", 0.60))
        self.defense_threshold = float(getattr(args, "mad_defense_threshold", 0.80))
        self.recovery_threshold = float(getattr(args, "mad_state_recovery_threshold", 0.45))
        self.history_threshold = float(getattr(args, "mad_history_threshold", 0.25))
        self.magnitude_threshold = float(getattr(args, "mad_magnitude_threshold", 0.70))
        if not 0 <= self.watch_threshold < self.suspicious_threshold < self.defense_threshold <= 1:
            raise ValueError("state thresholds require 0 <= watch < suspicious < defense <= 1")
        if not self.watch_threshold <= self.recovery_threshold < self.defense_threshold:
            raise ValueError("state recovery threshold must lie between watch and defense")

    def assess(self, observations, memory, round_number=0):
        clients = {}
        risks = []
        for cid, row in observations.items():
            prior = memory.get(cid)
            reputation = prior.reputation if prior is not None and self.use_reputation else 1.0
            historical = row["signals"]["temporal"] if self.use_temporal else 0.0
            evidence = bool(row.get("invalid_update")) or (
                row["anomaly"] >= self.anomaly_threshold and (not self.use_temporal or historical >= self.history_threshold)
            )
            _, streak, window_count, persistence = memory.evidence(cid, round_number, evidence)
            weights = dict(self.weights)
            if not self.use_reputation:
                weights["reputation"] = 0.0
            if not self.use_temporal:
                weights["history"] = weights["persistence"] = 0.0
            components = {"anomaly": row["anomaly"], "history": historical, "reputation": 1 - reputation,
                          "persistence": persistence * max(row["anomaly"], historical)}
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("ablation disables every nonzero risk weight")
            risk = _clip(sum(weights[name] * value for name, value in components.items()) / total)
            if row.get("invalid_update"):
                risk = 1.0
            repeated = streak >= self.high_patience or window_count >= self.high_patience + 1
            confirmed = risk >= self.defense_threshold and evidence and repeated
            if confirmed:
                state = "DEFENSE"
            elif prior is not None and prior.current_state == "DEFENSE" and risk >= self.recovery_threshold and evidence:
                state = "DEFENSE"
            elif risk >= self.suspicious_threshold:
                state = "SUSPICIOUS"
            elif risk >= self.watch_threshold or evidence:
                state = "WATCH"
            else:
                state = "NORMAL"
            clients[cid] = {"risk": risk, "level": self.level(risk), "state": state, "reputation_before": reputation,
                            "confirmed": confirmed, "high_streak": streak, "suspicious_in_window": window_count,
                            "persistence": persistence, "suspicious_evidence": evidence, "risk_components": components}
            risks.append(risk)
        count = len(risks)
        mean = sum(risks) / count if count else 0.0
        maximum = max(risks, default=0.0)
        variance = sum((risk - mean) ** 2 for risk in risks) / count if count else 0.0
        fraction = sum(risk >= self.suspicious_threshold for risk in risks) / count if count else 0.0
        round_components = {"mean": mean, "max": maximum, "fraction": fraction, "variance": min(1.0, 4 * variance)}
        round_risk = _clip(sum(self.round_weights[key] * value for key, value in round_components.items()))
        return {"clients": clients, "raw_scores": {cid: row["anomaly"] for cid, row in observations.items()},
                "round_risk": round_risk, "round_level": self.level(round_risk), "mean_risk": mean,
                "max_risk": maximum, "fraction_high": fraction, "risk_variance": variance,
                "high_risk_count": sum(risk >= self.defense_threshold for risk in risks),
                "max_anomaly": max((row["anomaly"] for row in observations.values()), default=0.0),
                "fraction_anomalous": sum(details["suspicious_evidence"] for details in clients.values()) / count if count else 0.0,
                "excessive_magnitude_fraction": sum(row["signals"]["norm"] >= self.magnitude_threshold and clients[cid]["suspicious_evidence"] for cid, row in observations.items()) / count if count else 0.0,
                "v2": True}
