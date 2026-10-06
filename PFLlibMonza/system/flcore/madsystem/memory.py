"""Compact per-client history and reversible reputation updates."""

from dataclasses import dataclass, field
import math


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
    client_id: int = -1
    historical_deviation_ema: float = 0.0
    current_state: str = "NORMAL"
    suspicious_window: list = field(default_factory=list)
    norm_variation_ema: float = 0.0
    cosine_variation_ema: float = 0.0
    distance_variation_ema: float = 0.0
    direction_ema: list = field(default_factory=list)


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


class HistoryAwareMemory(MemoryManager):
    """Bounded observation history; population diversity alone costs no reputation."""

    def __init__(self, alpha=0.9, penalty=0.08, recovery=0.02,
                 suspicious_threshold=0.45, window=5, initial_reputation=1.0,
                 suspicious_profile_weight=0.1):
        super().__init__(alpha, penalty, recovery, suspicious_threshold)
        if window < 1 or not 0 <= initial_reputation <= 1 or not 0 <= suspicious_profile_weight <= 1:
            raise ValueError("invalid memory window, initial reputation or profile weight")
        self.window = int(window)
        self.initial_reputation = float(initial_reputation)
        self.profile_weight = float(suspicious_profile_weight)

    def evidence(self, client_id, round_number, suspicious):
        state = self.get(client_id)
        recent = [] if state is None else [
            item for item in state.suspicious_window
            if round_number - self.window < item[0] < round_number
        ]
        recent.append((int(round_number), bool(suspicious)))
        consecutive = 0
        for index in range(len(recent) - 1, -1, -1):
            if not recent[index][1]:
                break
            if index < len(recent) - 1 and recent[index + 1][0] != recent[index][0] + 1:
                break
            consecutive += 1
        count = sum(item[1] for item in recent)
        # Denominator never collapses to one just because a client was absent.
        persistence = count / self.window
        return recent, consecutive, count, persistence

    def update(self, client_id, features, anomaly, risk, round_number,
               suspicious=False, current_state="NORMAL", use_reputation=True):
        state = self.clients.setdefault(client_id, ClientState(
            client_id=int(client_id), reputation=self.initial_reputation,
        ))
        first = state.observations == 0
        recent, streak, count, persistence = self.evidence(client_id, round_number, suspicious)
        retention = 1 - (1 - self.alpha) * (self.profile_weight if suspicious else 1)
        for name in ("norm", "cosine", "distance"):
            observed = float(features[name])
            old = getattr(state, f"{name}_ema")
            setattr(state, f"{name}_ema", observed if first else retention * old + (1 - retention) * observed)
            residual = float(features.get(f"_{name}_residual", 0.0))
            spread = getattr(state, f"{name}_variation_ema")
            setattr(state, f"{name}_variation_ema", retention * spread + (1 - retention) * residual)
        direction = features.get("_direction", [])
        if direction:
            previous = state.direction_ema
            combined = direction if not previous else [retention * old + (1 - retention) * new for old, new in zip(previous, direction)]
            norm = math.sqrt(sum(value * value for value in combined))
            state.direction_ema = [value / max(norm, 1e-12) for value in combined]
        state.historical_deviation_ema = self.alpha * state.historical_deviation_ema + (1 - self.alpha) * float(features["temporal"])
        state.anomaly_ema = float(anomaly) if first else self.alpha * state.anomaly_ema + (1 - self.alpha) * float(anomaly)
        if use_reputation:
            if suspicious and count >= 2:
                state.reputation = _clip(state.reputation - self.penalty * float(anomaly) * (0.5 + 0.5 * persistence))
            elif not suspicious:
                state.reputation = _clip(state.reputation + self.recovery * (1 - float(risk)))
        else:
            state.reputation = self.initial_reputation
        state.risk = _clip(risk)
        state.consecutive_suspicious_rounds = streak
        state.total_suspicious_rounds += int(suspicious)
        state.suspicious_window = recent
        state.current_state = str(current_state)
        state.last_seen_round = int(round_number)
        state.observations += 1
        return state
