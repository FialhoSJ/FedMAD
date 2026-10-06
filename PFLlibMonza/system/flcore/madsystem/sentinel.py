"""Continuous, deterministic inspection of client model updates."""

import math

import torch
import torch.nn.functional as F

from flcore.madsystem.monitoring import _updates, _direction_sketch
from flcore.madsystem.config import normalized_weights


def _high_tail(values):
    """Robust high-side score with a defined zero-spread behavior."""
    values = torch.nan_to_num(torch.as_tensor(values, dtype=torch.float64), nan=1e12, posinf=1e12, neginf=1e12)
    if values.numel() == 0:
        return []
    center = values.median()
    mad = (values - center).abs().median() * 1.4826
    scale = max(float(mad), 0.1 * abs(float(center)), 1e-6)
    z = ((values - center) / scale).clamp(min=0)
    return (1.0 - torch.exp(-z)).clamp(0, 1).tolist()


class FedMADSentinel:
    """One observer with magnitude, direction, population and temporal signals."""

    feature_names = ("norm", "cosine", "distance", "temporal")

    def inspect(self, client_ids, client_models, global_model, memory, use_temporal=True):
        if len(client_ids) != len(client_models):
            raise ValueError("client_ids and client_models must have equal length")
        if not client_ids:
            return {}
        _, matrix = _updates(client_models, global_model)
        matrix = torch.nan_to_num(matrix.double(), nan=1e12, posinf=1e12, neginf=-1e12)
        reference = matrix.median(dim=0).values
        norms = torch.linalg.vector_norm(matrix, dim=1)
        reference_norm = torch.linalg.vector_norm(reference)
        if reference_norm > 1e-12:
            cosines = F.cosine_similarity(matrix, reference.unsqueeze(0), dim=1, eps=1e-12)
        else:
            cosines = torch.where(norms < 1e-12, torch.ones_like(norms), torch.zeros_like(norms))
        distances = torch.linalg.vector_norm(matrix - reference, dim=1) / max(float(reference_norm), 1e-12)
        median_norm = norms.median()
        magnitude_deviation = (torch.log1p(norms) - torch.log1p(median_norm)).abs()
        norm_scores = _high_tail(magnitude_deviation)
        direction_scores = _high_tail((1.0 - cosines).clamp(0, 2))
        distance_scores = _high_tail(distances)
        rows = {}
        for index, cid in enumerate(client_ids):
            norm = float(norms[index])
            cosine = float(cosines[index])
            distance = float(distances[index])
            prior = memory.get(cid)
            if not use_temporal or prior is None or prior.observations == 0:
                temporal = 0.0
            else:
                norm_change = abs(math.log1p(norm) - math.log1p(prior.norm_ema))
                cosine_change = abs(cosine - prior.cosine_ema) / 2.0
                distance_change = abs(math.log1p(distance) - math.log1p(prior.distance_ema))
                temporal = min(1.0, 1.0 - math.exp(-(norm_change + cosine_change + distance_change)))
            signals = {
                "norm": float(norm_scores[index]),
                "cosine": float(direction_scores[index]),
                "distance": float(distance_scores[index]),
                "temporal": temporal,
            }
            anomaly = sum(signals.values()) / len(signals)
            rows[cid] = {
                "features": {"norm": norm, "cosine": cosine, "distance": distance, "temporal": temporal},
                "signals": signals,
                "anomaly": anomaly,
            }
        return rows


class HistoryAwareSentinel(FedMADSentinel):
    """Joint evidence from a robust population and each client's previous profile.

    A stable population outlier is not automatically suspicious. Raw population
    scores remain available for auditing, alongside the history-gated score.
    """

    def __init__(self, weights=None, population_floor=0.15, cold_start_factor=0.5,
                 min_observations=3, history_sensitivity=3.0, use_population=True):
        self.weights = normalized_weights(weights, {"population": 0.30, "history": 0.35, "norm": 0.20, "cosine": 0.15})
        if not 0 <= population_floor <= 1 or not 0 <= cold_start_factor <= 1:
            raise ValueError("population factors must be in [0,1]")
        if min_observations < 1 or not math.isfinite(history_sensitivity) or history_sensitivity <= 0:
            raise ValueError("history observations and sensitivity must be positive")
        self.population_floor = float(population_floor)
        self.cold_start_factor = float(cold_start_factor)
        self.min_observations = int(min_observations)
        self.sensitivity = float(history_sensitivity)
        self.use_population = bool(use_population)

    def inspect(self, client_ids, client_models, global_model, memory, use_temporal=True):
        if len(client_ids) != len(client_models) or len(set(client_ids)) != len(client_ids):
            raise ValueError("unique client IDs must match the uploaded models")
        if not client_ids:
            return {}
        _, matrix = _updates(client_models, global_model)
        invalid = ~torch.isfinite(matrix).all(dim=1)
        matrix = torch.nan_to_num(matrix.double(), nan=1e6, posinf=1e6, neginf=-1e6)
        reference = matrix.median(dim=0).values
        norms = torch.linalg.vector_norm(matrix, dim=1)
        reference_norm = torch.linalg.vector_norm(reference)
        cosines = F.cosine_similarity(matrix, reference.unsqueeze(0), dim=1, eps=1e-12)
        if reference_norm < 1e-12:
            cosines = torch.where(norms < 1e-12, torch.ones_like(norms), torch.zeros_like(norms))
        distances = torch.linalg.vector_norm(matrix - reference, dim=1) / max(float(reference_norm), 1e-6)
        norm_scores = _high_tail((torch.log1p(norms) - torch.log1p(norms.median())).abs())
        cosine_scores = _high_tail((1 - cosines).clamp(0, 2))
        population_scores = _high_tail(distances)
        changes = {}
        directions = {}
        for index, cid in enumerate(client_ids):
            directions[cid] = _direction_sketch(matrix[index], size=64).tolist()
            prior = memory.get(cid)
            if use_temporal and prior is not None and prior.observations >= self.min_observations:
                changes[cid] = [
                    math.log1p(float(norms[index])) - math.log1p(prior.norm_ema),
                    (float(cosines[index]) - prior.cosine_ema) / 2,
                    math.log1p(float(distances[index])) - math.log1p(prior.distance_ema),
                ]
        # Common training drift is evidence about the population, not an attack label.
        common = torch.tensor(list(changes.values())).median(dim=0).values.tolist() if len(changes) >= 3 else [0.0] * 3
        rows = {}
        for index, cid in enumerate(client_ids):
            prior = memory.get(cid)
            mature = cid in changes
            residuals = [0.0] * 3
            historical = 0.0
            if mature:
                residuals = [abs(value - drift) for value, drift in zip(changes[cid], common)]
                scales = [max(0.1, prior.norm_variation_ema), max(0.05, prior.cosine_variation_ema), max(0.1, prior.distance_variation_ema)]
                normalized = [residuals[0] / scales[0]]
                if self.use_population:
                    normalized.extend([residuals[1] / scales[1], residuals[2] / scales[2]])
                if prior.direction_ema:
                    dot = sum(old * new for old, new in zip(prior.direction_ema, directions[cid]))
                    normalized.append(max(0.0, min(1.0, (1 - dot) / 2)) / 0.1)
                historical = min(1.0, 1 - math.exp(-sum(normalized) / max(1, len(normalized)) / self.sensitivity))
            signals = {"norm": norm_scores[index], "cosine": cosine_scores[index], "distance": population_scores[index], "temporal": historical}
            raw = {"population": signals["distance"], "history": historical, "norm": signals["norm"], "cosine": signals["cosine"]}
            enabled = dict(self.weights)
            if not use_temporal:
                enabled["history"] = 0.0
            if not self.use_population:
                for key in ("population", "norm", "cosine"):
                    enabled[key] = 0.0
            weight_sum = sum(enabled.values())
            if weight_sum <= 0:
                raise ValueError("ablation disables every nonzero anomaly weight")
            gate = (self.population_floor + (1 - self.population_floor) * historical if mature else self.cold_start_factor) if use_temporal else 1.0
            effective = {key: value if key == "history" else gate * value for key, value in raw.items()}
            anomaly = sum(enabled[key] * effective[key] for key in enabled) / weight_sum
            if invalid[index]:
                anomaly = 1.0
            rows[cid] = {
                "features": {"norm": float(norms[index]), "cosine": float(cosines[index]), "distance": float(distances[index]), "temporal": historical,
                             "_norm_residual": residuals[0], "_cosine_residual": residuals[1], "_distance_residual": residuals[2], "_direction": directions[cid]},
                "signals": signals, "anomaly": min(1.0, max(0.0, anomaly)),
                "population_anomaly": (raw["population"] + raw["norm"] + raw["cosine"]) / 3,
                "history_mature": mature, "population_gate": gate,
                "invalid_update": bool(invalid[index]),
            }
        return rows
