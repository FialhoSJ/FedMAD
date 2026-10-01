"""Independent robust aggregation mechanisms used by the FedMAD meta-agent."""

import copy

import torch
import torch.nn.functional as F

from flcore.madsystem.monitoring import _direction_sketch, _updates


def _normalized_weights(weights, count, device):
    if count == 0:
        return torch.empty(0, device=device)
    values = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if values.numel() != count or not torch.isfinite(values).all() or values.sum() <= 0:
        values = torch.ones(count, dtype=torch.float32, device=device)
    values = values.clamp(min=0)
    if values.sum() <= 0:
        values = torch.ones_like(values)
    return values / values.sum()


def _weighted_coordinate_mean(updates, weights):
    weights = _normalized_weights(weights, updates.shape[0], updates.device)
    return (updates * weights.view(-1, *([1] * (updates.ndim - 1)))).sum(dim=0)


class DefenseAgent:
    name = "defense"

    def aggregate(
        self,
        server_model,
        client_models,
        weights,
        client_ids,
        risk_scores=None,
        byzantine_f=1,
        clip_norm=1.0,
    ):
        raise NotImplementedError

    @staticmethod
    def _apply_update(server_model, client_models, reducer):
        if not client_models:
            return copy.deepcopy(server_model)
        candidate = copy.deepcopy(server_model)
        base_state = server_model.state_dict()
        client_states = [model.state_dict() for model in client_models]
        with torch.no_grad():
            candidate_state = candidate.state_dict()
            for name, target in candidate_state.items():
                reference = base_state[name]
                if not reference.is_floating_point():
                    continue
                deltas = torch.stack([
                    (state[name].detach().to(reference.device, dtype=torch.float32)
                     - reference.detach().float())
                    for state in client_states
                ])
                aggregate_delta = reducer(deltas)
                target.copy_((reference.detach().float() + aggregate_delta).to(target.dtype))
        return candidate


class FedAvgDefense(DefenseAgent):
    name = "fedavg"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, **kwargs):
        device = next(server_model.parameters()).device
        normalized = _normalized_weights(weights, len(client_models), device)
        return self._apply_update(
            server_model,
            client_models,
            lambda updates: _weighted_coordinate_mean(updates, normalized),
        )


class CoordinateMedianDefense(DefenseAgent):
    name = "median"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, **kwargs):
        return self._apply_update(
            server_model, client_models, lambda updates: updates.median(dim=0).values
        )


class TrimmedMeanDefense(DefenseAgent):
    name = "trimmed_mean"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, byzantine_f=1, **kwargs):
        count = len(client_models)
        trim = min(max(0, int(byzantine_f)), max(0, (count - 1) // 2))

        def reduce(updates):
            ordered = updates.sort(dim=0).values
            selected = ordered[trim:count - trim] if trim else ordered
            return selected.mean(dim=0)

        return self._apply_update(server_model, client_models, reduce)


class ClippingDefense(DefenseAgent):
    name = "clipping"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, clip_norm=1.0, **kwargs):
        device = next(server_model.parameters()).device
        coefficients = _normalized_weights(weights, len(client_models), device)

        def reduce(updates):
            norms = torch.linalg.vector_norm(updates.reshape(updates.shape[0], -1), dim=1)
            factors = (float(clip_norm) / (norms + 1e-12)).clamp(max=1.0)
            clipped = updates * factors.view(-1, *([1] * (updates.ndim - 1)))
            return _weighted_coordinate_mean(clipped, coefficients)

        return self._apply_update(server_model, client_models, reduce)


def _krum_scores(vectors, byzantine_f):
    count = vectors.shape[0]
    if count < 2:
        return torch.zeros(count, dtype=torch.float32)
    f = min(max(0, int(byzantine_f)), max(0, (count - 3) // 2))
    neighbor_count = max(1, min(count - f - 2, count - 1))
    distances = torch.cdist(vectors.float(), vectors.float(), p=2).pow(2)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.topk(distances, k=neighbor_count, largest=False, dim=1).values
    return nearest.sum(dim=1)


class KrumDefense(DefenseAgent):
    name = "krum"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, byzantine_f=1, **kwargs):
        if len(client_models) < 3:
            return FedAvgDefense().aggregate(server_model, client_models, weights, client_ids)
        _, vectors = _updates(client_models, server_model)
        chosen = int(_krum_scores(vectors, byzantine_f).argmin())
        return self._apply_update(
            server_model, client_models, lambda updates: updates[chosen]
        )


class MultiKrumDefense(DefenseAgent):
    name = "multi_krum"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, byzantine_f=1, **kwargs):
        count = len(client_models)
        if count < 3:
            return FedAvgDefense().aggregate(server_model, client_models, weights, client_ids)
        _, vectors = _updates(client_models, server_model)
        f = min(max(0, int(byzantine_f)), max(0, (count - 3) // 2))
        selected_count = max(1, count - f - 2)
        selected = torch.topk(_krum_scores(vectors, f), k=selected_count, largest=False).indices.tolist()
        selected_models = [client_models[index] for index in selected]
        selected_weights = [weights[index] for index in selected]
        return FedAvgDefense().aggregate(
            server_model, selected_models, selected_weights,
            [client_ids[index] for index in selected],
        )


class BulyanDefense(DefenseAgent):
    name = "bulyan"

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, byzantine_f=1, **kwargs):
        count = len(client_models)
        f = max(0, int(byzantine_f))
        if count < 4 * f + 3 or f == 0:
            return TrimmedMeanDefense().aggregate(
                server_model, client_models, weights, client_ids, byzantine_f=f
            )
        _, vectors = _updates(client_models, server_model)
        candidate_count = count - 2 * f
        remaining_vectors = vectors.float()
        remaining_indices = torch.arange(count)
        selected_indices = []
        while len(selected_indices) < candidate_count:
            local_index = int(_krum_scores(remaining_vectors, f).argmin())
            selected_indices.append(int(remaining_indices[local_index]))
            keep = torch.ones(
                remaining_indices.shape[0], dtype=torch.bool
            )
            keep[local_index] = False
            remaining_vectors = remaining_vectors[keep]
            remaining_indices = remaining_indices[keep]
        candidates = torch.as_tensor(selected_indices, dtype=torch.long)
        beta = max(1, count - 4 * f)

        def reduce(updates):
            selected = updates[candidates.to(updates.device)]
            median = selected.median(dim=0).values
            distances = (selected - median).abs()
            closest = torch.topk(
                distances, k=min(beta, selected.shape[0]), largest=False, dim=0
            ).indices
            values = selected.gather(0, closest)
            return values.mean(dim=0)

        return self._apply_update(server_model, client_models, reduce)


class FoolsGoldDefense(DefenseAgent):
    name = "foolsgold"

    def __init__(self):
        self.client_memory = {}

    def observe(self, client_models, server_model, client_ids):
        _, vectors = _updates(client_models, server_model)
        for cid, vector in zip(client_ids, vectors):
            current = _direction_sketch(vector.detach().cpu())
            previous = self.client_memory.get(cid)
            self.client_memory[cid] = current if previous is None else previous + current

    def aggregate(self, server_model, client_models, weights, client_ids, risk_scores=None, **kwargs):
        if len(client_models) < 2:
            return FedAvgDefense().aggregate(server_model, client_models, weights, client_ids)
        _, current = _updates(client_models, server_model)
        vectors = []
        for cid, vector in zip(client_ids, current):
            sketch = _direction_sketch(vector.detach().cpu())
            vectors.append(self.client_memory.get(cid, sketch).float())
        directions = F.normalize(torch.stack(vectors), p=2, dim=1, eps=1e-12)
        similarities = directions @ directions.T
        similarities.fill_diagonal_(0.0)
        max_similarity = similarities.max(dim=1).values.clamp(min=0, max=1)
        # Rescale penalties across this round; identical clients still get equal weight.
        if max_similarity.max() - max_similarity.min() > 1e-8:
            max_similarity = (max_similarity - max_similarity.min()) / (
                max_similarity.max() - max_similarity.min()
            )
        gold = (1.0 - max_similarity).clamp(min=1e-3)
        device = next(server_model.parameters()).device
        base = _normalized_weights(weights, len(client_models), device)
        combined = _normalized_weights(
            base * gold.to(base.device), len(client_models), base.device
        )
        return self._apply_update(
            server_model,
            client_models,
            lambda updates: _weighted_coordinate_mean(
                updates.to(combined.device), combined
            ).to(updates.device),
        )


def build_defenses():
    """Create fresh stateful defense agents for one server run."""
    return {
        "fedavg": FedAvgDefense(),
        "median": CoordinateMedianDefense(),
        "trimmed_mean": TrimmedMeanDefense(),
        "clipping": ClippingDefense(),
        "krum": KrumDefense(),
        "multi_krum": MultiKrumDefense(),
        "bulyan": BulyanDefense(),
        "foolsgold": FoolsGoldDefense(),
    }
