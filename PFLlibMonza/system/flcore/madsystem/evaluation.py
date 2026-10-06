"""Simulation-only ground truth and metrics; never consumed by defense agents."""

from statistics import mean

import torch
import torch.nn.functional as F

from utils.data_utils import read_data


def confusion_metrics(predictions, labels):
    if len(predictions) != len(labels):
        raise ValueError("predictions and truth must align")
    tp = sum(bool(p) and bool(y) for p, y in zip(predictions, labels))
    fp = sum(bool(p) and not bool(y) for p, y in zip(predictions, labels))
    fn = sum(not bool(p) and bool(y) for p, y in zip(predictions, labels))
    tn = sum(not bool(p) and not bool(y) for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "tpr": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0}


class ExperimentEvaluator:
    """Observe completed decisions and actual uploads, including censored attacks."""

    def __init__(self, controlled_ids=()):
        self.controlled_ids = set(map(int, controlled_ids))
        self.episodes = []
        self.current_episodes = {}
        self.rounds = []

    def record_round(self, round_number, client_ids, predictions, active_attacks,
                     attack_types=None, robust_defense=False):
        attack_types = attack_types or {}
        truth = [bool(active_attacks.get(cid, False)) for cid in client_ids]
        detection = confusion_metrics(predictions, truth)
        for cid, detected, active in zip(client_ids, predictions, truth):
            attack = attack_types.get(cid, "unspecified")
            previous = self.current_episodes.get(cid)
            if not active:
                self.current_episodes.pop(cid, None)
                continue
            if previous is None or previous["attack"] != attack:
                previous = {"client_id": int(cid), "attack": attack, "start_round": int(round_number), "last_active_round": int(round_number), "detected_round": None, "latency": None}
                self.episodes.append(previous)
                self.current_episodes[cid] = previous
            previous["last_active_round"] = int(round_number)
            if detected and previous["detected_round"] is None:
                previous["detected_round"] = int(round_number)
                previous["latency"] = int(round_number) - previous["start_round"]
        self.rounds.append({"detection": detection, "eligible": bool(client_ids),
                            "benign": bool(client_ids) and not any(truth), "activated": bool(robust_defense)})
        return {"detection": detection,
                "malicious_ground_truth": [int(cid in self.controlled_ids) for cid in client_ids],
                "active_attack_ground_truth": list(map(int, truth)),
                "attack_types": {str(cid): attack_types.get(cid) for cid in client_ids if active_attacks.get(cid)},
                "evaluation_summary": self.summary(include_episodes=False)}

    def summary(self, include_episodes=True):
        eligible = [row for row in self.rounds if row["eligible"]]
        benign = [row for row in eligible if row["benign"]]
        observed = [episode["latency"] for episode in self.episodes if episode["latency"] is not None]
        result = {"defense_activation_rate": sum(row["activated"] for row in eligible) / len(eligible) if eligible else None,
                "unnecessary_defense_rate": sum(row["activated"] for row in benign) / len(benign) if benign else None,
                "benign_rounds": len(benign), "attack_episodes": len(self.episodes),
                "detected_attack_episodes": len(observed), "detection_latency": mean(observed) if observed else None,
                "latency_censored_fraction": 1 - len(observed) / len(self.episodes) if self.episodes else None}
        if include_episodes:
            result["latency_by_client_attack"] = [dict(episode) for episode in self.episodes]
        return result

    def record_server_round(self, round_number, client_ids, predictions, clients, robust_defense=False, risk_predictions=None):
        """Only this simulation module reads actual malicious-upload flags."""
        active = {cid: bool(getattr(clients[cid], "is_malicious", False)) for cid in client_ids}
        attacks = {cid: getattr(clients[cid], "last_attack", "unspecified") for cid in client_ids}
        result = self.record_round(round_number, client_ids, predictions, active, attacks, robust_defense)
        if risk_predictions is not None:
            result["risk_detection"] = confusion_metrics(risk_predictions, [active[cid] for cid in client_ids])
        return result

    @staticmethod
    def evaluate_model(model, dataset, num_clients, batch_size=32, source_label=None, target_label=None, is_train=False):
        """Global model on clean held-out test data, streamed one client at a time."""
        if batch_size < 1:
            raise ValueError("evaluation batch size must be positive")
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        examples = correct = target_examples = target_correct = 0
        total_loss = 0.0
        try:
            with torch.no_grad():
                for cid in range(num_clients):
                    data = read_data(dataset, cid, is_train=is_train)
                    for start in range(0, len(data["y"]), batch_size):
                        x = torch.as_tensor(data["x"][start:start + batch_size], dtype=torch.float32, device=device)
                        y = torch.as_tensor(data["y"][start:start + batch_size], dtype=torch.long, device=device)
                        logits = model(x)
                        predictions = logits.argmax(dim=1)
                        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
                        correct += int((predictions == y).sum())
                        examples += len(y)
                        if source_label is not None and target_label is not None:
                            source = y == source_label
                            target_examples += int(source.sum())
                            target_correct += int((source & (predictions == target_label)).sum())
        finally:
            model.train(was_training)
        return {"accuracy": correct / examples if examples else None, "test_loss": total_loss / examples if examples else None,
                "examples": examples, "attack_success_rate": target_correct / target_examples if target_examples else None,
                "asr_examples": target_examples,
                "asr_definition": "source-class clean examples classified as target under targeted label poisoning" if source_label is not None and target_label is not None else "not applicable: no configured targeted attack"}


def evaluate_global(server, acc=None, loss=None):
    """V2 evaluation hook: global parameters, not post-training local models."""
    args = server.args
    batch_size = int(getattr(args, "mad_evaluation_batch_size", 32))
    targeted = args.atack in ("label", "label_flipping")
    metrics = ExperimentEvaluator.evaluate_model(
        server.global_model, server.dataset, server.num_clients, batch_size,
        getattr(args, "attack_source_label", None) if targeted else None,
        getattr(args, "attack_target_label", None) if targeted else None,
    )
    train = ExperimentEvaluator.evaluate_model(server.global_model, server.dataset, server.num_clients, batch_size, is_train=True)
    server.rs_test_acc.append(metrics["accuracy"])
    server.rs_train_loss.append(train["test_loss"])
    server.rs_test_auc.append(0.0)  # Auxiliary AUC is not evaluated by this global hook.
    server.latest_global_metrics = {**metrics, "train_loss": train["test_loss"],
                                   "clean_accuracy": metrics["accuracy"],
                                   "robust_accuracy": metrics["accuracy"] if server.n_client_malicious else None,
                                   "evaluation_position": "after_aggregation"}
    print(f"[V2 global evaluation] accuracy={metrics['accuracy']:.4f} train_loss={train['test_loss']:.4f}")
    return server.latest_global_metrics
