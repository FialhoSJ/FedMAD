"""Compact per-client history and reversible reputation updates."""

from dataclasses import dataclass


def _clip(value):
    return min(1.0, max(0.0, float(value)))


@dataclass
class ClientState:
    norm_ema: float = 0.0
    cosine_ema: float = 0.0
    distance_ema: float = 0.0
    anomaly_ema: float = 0.0
    reputation: float = 1.0
    risk: float = 0.0
    consecutive_suspicious_rounds: int = 0
    total_suspicious_rounds: int = 0
    last_seen_round: int = -1
    observations: int = 0


class MemoryManager:
    def __init__(self, alpha=0.9, penalty=0.08, recovery=0.02, suspicious_threshold=0.65):
        if not 0 < alpha < 1:
            raise ValueError("history alpha must be between 0 and 1")
        if not 0 <= recovery < penalty <= 1:
            raise ValueError("require 0 <= recovery < penalty <= 1")
        self.alpha = float(alpha)
        self.penalty = float(penalty)
        self.recovery = float(recovery)
        self.suspicious_threshold = _clip(suspicious_threshold)
        self.clients = {}

    def get(self, client_id):
        return self.clients.get(client_id)

    def update(self, client_id, features, anomaly, risk, round_number):
        state = self.clients.setdefault(client_id, ClientState())
        first = state.observations == 0
        for name in ("norm", "cosine", "distance"):
            observed = float(features[name])
            old = getattr(state, f"{name}_ema")
            setattr(state, f"{name}_ema", observed if first else self.alpha * old + (1 - self.alpha) * observed)
        anomaly = _clip(anomaly)
        state.anomaly_ema = anomaly if first else self.alpha * state.anomaly_ema + (1 - self.alpha) * anomaly
        state.reputation = _clip(state.reputation - self.penalty * anomaly + self.recovery * (1 - anomaly))
        state.risk = _clip(risk)
        suspicious = anomaly >= self.suspicious_threshold
        state.consecutive_suspicious_rounds = state.consecutive_suspicious_rounds + 1 if suspicious else 0
        state.total_suspicious_rounds += int(suspicious)
        state.last_seen_round = int(round_number)
        state.observations += 1
        return state
