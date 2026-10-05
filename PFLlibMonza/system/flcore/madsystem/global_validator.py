"""Candidate-model validator with a trusted-model rollback reference."""

import math

import torch
import torch.nn.functional as F


class GlobalValidator:
    def __init__(self, clients, args, device):
        self.device = torch.device(device)
        self.batch_limit = max(1, int(getattr(args, "mad_validation_batches", 2)))
        self.client_limit = max(0, int(getattr(args, "mad_validation_clients", 12)))
        self.loss_tolerance = max(0.0, float(getattr(args, "mad_validation_loss_tolerance", 0.25)))
        self.accuracy_tolerance = max(0.0, float(getattr(args, "mad_validation_accuracy_tolerance", 0.10)))
        self.max_delta_norm = float(getattr(args, "mad_validation_max_delta", 50.0))
        ordered_clients = sorted(clients, key=lambda client: getattr(client, "id", 0))
        count = self.client_limit or len(ordered_clients)
        self.validation_clients = self._representative_clients(ordered_clients, count)
        self.client_batches = {}
        self.validation_batches = []
        for client in self.validation_clients:
            batches = self.capture_batches(client)
            self.client_batches[client.id] = batches
            if batches:
                self.validation_batches.extend(batches)
        self.trusted_metrics = None

    @staticmethod
    def _representative_clients(clients, count):
        """Pick a deterministic, evenly spaced subset across client IDs."""
        if count <= 0 or count >= len(clients):
            return list(clients)
        if count == 1:
            return [clients[(len(clients) - 1) // 2]]
        last_index = len(clients) - 1
        return [
            clients[round(index * last_index / (count - 1))]
            for index in range(count)
        ]

    def capture_batches(self, client):
        """Capture a small, fixed local test subset for repeatable candidate checks."""
        batches = []
        try:
            loader = client.load_test_data()
            for index, batch in enumerate(loader):
                if index >= self.batch_limit:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                    if isinstance(x, (list, tuple)):
                        x = x[0]
                    batches.append((x.detach().cpu(), y.detach().cpu()))
        except (OSError, ValueError, RuntimeError, TypeError):
            return []
        return batches

    def evaluate(self, model, batches=None):
        batches = self.validation_batches if batches is None else batches
        if not batches:
            return {"available": False, "loss": None, "accuracy": None, "examples": 0}
        was_training = model.training
        model.eval()
        total_loss = 0.0
        correct = 0
        examples = 0
        try:
            with torch.no_grad():
                for x, y in batches:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = model(x)
                    loss = F.cross_entropy(output, y, reduction="sum")
                    total_loss += float(loss.item())
                    correct += int((output.argmax(dim=1) == y).sum().item())
                    examples += int(y.shape[0])
        finally:
            model.train(was_training)
        if examples == 0:
            return {"available": False, "loss": None, "accuracy": None, "examples": 0}
        return {
            "available": True,
            "loss": total_loss / examples,
            "accuracy": correct / examples,
            "examples": examples,
        }

    def loss_impact(self, candidate, reference, batches):
        """Positive loss change on the same client's held-out examples."""
        current = self.evaluate(candidate, batches)
        baseline = self.evaluate(reference, batches)
        if not current["available"] or not baseline["available"]:
            return 0.0
        if not math.isfinite(current["loss"]) or not math.isfinite(baseline["loss"]):
            return float("inf")
        return current["loss"] - baseline["loss"]

    @staticmethod
    def _all_finite(model):
        for value in model.state_dict().values():
            if value.is_floating_point() and not torch.isfinite(value).all():
                return False
        return True

    @staticmethod
    def _delta_norm(candidate, reference):
        total = 0.0
        candidate_params = dict(candidate.named_parameters())
        for name, parameter in reference.named_parameters():
            if name in candidate_params:
                delta = candidate_params[name].detach().float() - parameter.detach().float()
                total += float(torch.sum(delta * delta).item())
        return math.sqrt(total)

    def validate(self, candidate, trusted_model):
        if not self._all_finite(candidate):
            return False, "non_finite_parameters", {
                "available": False, "loss": None, "accuracy": None, "examples": 0
            }

        delta_norm = self._delta_norm(candidate, trusted_model)
        try:
            metrics = self.evaluate(candidate)
            baseline = self.trusted_metrics
            if baseline is None:
                baseline = self.evaluate(trusted_model)
        except (RuntimeError, ValueError, TypeError):
            return False, "validation_error", {
                "available": False,
                "loss": None,
                "accuracy": None,
                "examples": 0,
                "delta_norm": delta_norm,
            }
        metrics["delta_norm"] = delta_norm
        if delta_norm > self.max_delta_norm:
            return False, "update_norm_exceeded", metrics

        if metrics["available"] and (
            not math.isfinite(metrics["loss"])
            or not math.isfinite(metrics["accuracy"])
        ):
            return False, "non_finite_validation_metrics", metrics

        if metrics["available"] and baseline["available"]:
            loss_limit = baseline["loss"] * (1.0 + self.loss_tolerance) + 1e-8
            accuracy_floor = baseline["accuracy"] - self.accuracy_tolerance
            if metrics["loss"] > loss_limit:
                return False, "validation_loss_regression", metrics
            if metrics["accuracy"] < accuracy_floor:
                return False, "validation_accuracy_regression", metrics
        return True, "accepted", metrics

    def commit(self, metrics):
        if metrics and metrics.get("available"):
            self.trusted_metrics = {
                "available": True,
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "examples": int(metrics.get("examples", 0)),
            }
