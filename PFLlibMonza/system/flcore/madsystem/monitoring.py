"""Monitoring agents and temporal risk assessment for FedMAD."""

from collections import defaultdict
import math

import torch
import torch.nn.functional as F


def _updates(client_models, global_model):
    """Return floating state deltas and one flattened update per client."""
    global_state = global_model.state_dict()
    named_updates = []
    flat_updates = []
    for model in client_models:
        client_state = model.state_dict()
        current = {}
        pieces = []
        for name, base in global_state.items():
            if (
                not base.is_floating_point()
                or name not in client_state
                or client_state[name].shape != base.shape
            ):
                continue
            delta = client_state[name].detach().float().cpu() - base.detach().float().cpu()
            current[name] = delta
            pieces.append(delta.reshape(-1))
        named_updates.append(current)
        flat_updates.append(torch.cat(pieces) if pieces else torch.zeros(1))
    return named_updates, torch.stack(flat_updates) if flat_updates else torch.empty(0, 1)


def _robust_high_scores(values):
    """Map high-side robust outliers to [0, 1] without min-max instability."""
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        return []
    values = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    median = values.median()
    mad = (values - median).abs().median() * 1.4826
    if mad < 1e-8:
        spread = values.std(unbiased=False)
        if spread < 1e-8:
            return [0.0] * values.numel()
        z = ((values - median) / (spread + 1e-8)).clamp(min=0)
    else:
        z = ((values - median) / mad).clamp(min=0)
    return (1.0 - torch.exp(-z / 3.0)).clamp(0, 1).tolist()


def _direction_sketch(vector, size=2048):
    size = min(int(size), max(1, vector.numel()))
    sketch = F.adaptive_avg_pool1d(
        vector.reshape(1, 1, -1), output_size=size
    ).reshape(-1)
    return F.normalize(sketch, p=2, dim=0, eps=1e-12)


class MonitoringAgent:
    name = "Monitoring"

    def analyze(self, client_models, global_model, metadata):
        raise NotImplementedError


class GradientAgent(MonitoringAgent):
    """Detect unusually large client update norms."""

    name = "Gradient"

    def analyze(self, client_models, global_model, metadata):
        _, matrix = _updates(client_models, global_model)
        if matrix.shape[0] < 2:
            return [0.0] * matrix.shape[0]
        return _robust_high_scores(torch.linalg.vector_norm(matrix, dim=1))


class SimilarityAgent(MonitoringAgent):
    """Score updates that point away from the other participating clients."""

    name = "Similarity"

    def analyze(self, client_models, global_model, metadata):
        _, matrix = _updates(client_models, global_model)
        n = matrix.shape[0]
        if n < 2:
            return [0.0] * n
        normalized = F.normalize(matrix, p=2, dim=1, eps=1e-12)
        similarity = normalized @ normalized.T
        distances = 1.0 - (similarity.sum(dim=1) - similarity.diag()) / (n - 1)
        return _robust_high_scores(distances)


class StatisticalAgent(MonitoringAgent):
    """Find clients with unusual update distributions across model layers."""

    name = "Statistical"

    def analyze(self, client_models, global_model, metadata):
        named, matrix = _updates(client_models, global_model)
        if matrix.shape[0] < 2:
            return [0.0] * matrix.shape[0]
        feature_rows = []
        for update in named:
            layer_norms = torch.stack(
                [torch.linalg.vector_norm(value) for value in update.values()]
            ) if update else torch.zeros(1)
            flat = torch.cat([value.reshape(-1) for value in update.values()]) if update else torch.zeros(1)
            feature_rows.append(torch.stack((
                flat.abs().mean(),
                flat.std(unbiased=False),
                flat.abs().max(),
                torch.log1p(torch.linalg.vector_norm(flat)),
                layer_norms.std(unbiased=False),
            )))
        features = torch.stack(feature_rows)
        center = features.median(dim=0).values
        scale = (features - center).abs().median(dim=0).values * 1.4826
        scale = torch.where(scale < 1e-8, features.std(dim=0, unbiased=False) + 1e-8, scale)
        distances = torch.linalg.vector_norm((features - center) / scale, dim=1)
        return _robust_high_scores(distances)


class PerformanceAgent(MonitoringAgent):
    """Use each update's measured validation-loss impact as an anomaly signal."""

    name = "Performance"

    def analyze(self, client_models, global_model, metadata):
        client_ids = metadata.get("client_ids", [])
        impacts = metadata.get("performance_impacts", {})
        values = [float(impacts.get(cid, 0.0)) for cid in client_ids]
        return _robust_high_scores(values)


class HistoryAgent(MonitoringAgent):
    """Compare updates with a client's recent directions, with a short memory."""

    name = "History"

    def __init__(self, lookback=5):
        self.lookback = max(1, int(lookback))
        self.client_updates = defaultdict(list)

    def analyze(self, client_models, global_model, metadata):
        client_ids = metadata.get("client_ids", [])
        _, matrix = _updates(client_models, global_model)
        if matrix.shape[0] == 0:
            return []
        scores = []
        for cid, update in zip(client_ids, matrix):
            history = self.client_updates[cid]
            current = _direction_sketch(update)
            if history:
                previous_raw = torch.stack(history).mean(dim=0)
                previous = F.normalize(previous_raw, p=2, dim=0, eps=1e-12)
                if (
                    torch.linalg.vector_norm(update) < 1e-12
                    or torch.linalg.vector_norm(previous_raw) < 1e-12
                ):
                    scores.append(0.0)
                else:
                    scores.append(float(((1.0 - torch.dot(previous, current)) / 2.0).clamp(0, 1)))
            else:
                scores.append(0.0)
            history.append(current.detach())
            if len(history) > self.lookback:
                del history[0]
        return scores


class RiskAssessment:
    """Fuse current monitoring signals with per-client temporal memory."""

    def __init__(self, args):
        self.low_threshold = min(
            1.0, max(0.0, float(getattr(args, "mad_low_threshold", 0.35)))
        )
        self.high_threshold = min(
            1.0,
            max(self.low_threshold, float(getattr(args, "mad_high_threshold", 0.65))),
        )
        self.current_weight = min(
            1.0, max(0.0, float(getattr(args, "mad_ema_current_weight", 0.45)))
        )
        self.quarantine_patience = max(1, int(getattr(args, "mad_high_patience", 2)))
        self.profiles = {}

    @staticmethod
    def _level(score, low, high):
        if score >= high:
            return "HIGH"
        if score >= low:
            return "MEDIUM"
        return "LOW"

    def assess(self, client_ids, client_scores):
        per_client = {}
        raw_scores = {}
        for cid in client_ids:
            signals = []
            for value in client_scores.get(cid, []):
                value = float(value)
                signals.append(min(1.0, max(0.0, value)) if math.isfinite(value) else 1.0)
            raw = sum(signals) / len(signals) if signals else 0.0
            profile = self.profiles.setdefault(cid, {
                "risk": 0.0,
                "observations": 0,
                "high_streak": 0,
                "alerts": 0,
            })
            risk = (
                self.current_weight * raw
                + (1.0 - self.current_weight) * profile["risk"]
            )
            level = self._level(risk, self.low_threshold, self.high_threshold)
            profile["risk"] = risk
            profile["observations"] += 1
            profile["alerts"] += int(raw >= self.high_threshold)
            profile["high_streak"] = profile["high_streak"] + 1 if level == "HIGH" else 0
            raw_scores[cid] = raw
            per_client[cid] = {
                "risk": risk,
                "level": level,
                "high_streak": profile["high_streak"],
                "observations": profile["observations"],
                "alerts": profile["alerts"],
                "confirmed": profile["high_streak"] >= self.quarantine_patience,
            }

        if per_client:
            risks = sorted((value["risk"] for value in per_client.values()), reverse=True)
            # A small attacker cohort must still affect the round policy.
            # Temporal smoothing prevents one unusual Non-IID update from
            # immediately producing a HIGH round.
            round_risk = risks[0]
        else:
            round_risk = 0.0
        return {
            "raw_scores": raw_scores,
            "clients": per_client,
            "round_risk": round_risk,
            "round_level": self._level(round_risk, self.low_threshold, self.high_threshold),
        }
