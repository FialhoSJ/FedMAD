"""Deterministic client and population risk estimates."""

from flcore.madsystem.memory import _clip


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
