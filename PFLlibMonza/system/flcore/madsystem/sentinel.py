"""Continuous, deterministic inspection of client model updates."""

import math

import torch
import torch.nn.functional as F

from flcore.madsystem.monitoring import _updates


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
