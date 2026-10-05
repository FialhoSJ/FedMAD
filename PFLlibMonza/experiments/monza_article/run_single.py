#!/usr/bin/env python3
"""Run one instrumented, article-parameter MONZA/FedMAD/FedAvg experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import runpy
import shutil
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SYSTEM = REPO / "system"
ATTACK_TYPES = ("zero", "random", "shuffle", "label")


def attack_seed(seed: int, round_number: int, client_id: int) -> int:
    return (
        int(seed) * 1_000_003
        + (int(round_number) + 1) * 100_003
        + int(client_id) * 1_009
        + 7_919
    ) % (2**63 - 1)


def selected_attack(seed: int, round_number: int, client_id: int, rate: float = 1.0):
    rng = random.Random(attack_seed(seed, round_number, client_id))
    if rng.random() >= float(rate):
        return None
    return rng.choice(ATTACK_TYPES)


def planned_attack_schedule(seed: int, malicious_ids: set[int], rounds: int):
    return [
        {
            "round": round_number,
            "client_id": client_id,
            "attack": selected_attack(seed, round_number, client_id),
        }
        for round_number in range(1, rounds + 1)
        for client_id in sorted(malicious_ids)
    ]


def confusion_rates(malicious_ids: set[int], blocked_ids: set[int], total: int):
    benign_total = total - len(malicious_ids)
    true_positive = len(malicious_ids & blocked_ids)
    false_positive = len(blocked_ids - malicious_ids)
    false_negative = len(malicious_ids - blocked_ids)
    true_negative = benign_total - false_positive
    return {
        "fpr": false_positive / benign_total if benign_total else None,
        "frr": false_negative / len(malicious_ids) if malicious_ids else None,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "blocked_clients": sorted(blocked_ids),
    }


def model_forward_macs_per_example(model, dataset_name: str) -> int:
    import torch
    from torch import nn

    model = copy.deepcopy(model).to("cpu")
    height = 28 if "MNIST" in dataset_name else 32
    channels = 1 if "MNIST" in dataset_name else 3
    total_macs = 0
    hooks = []

    def count_layer(module, inputs, output):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            out = output
            kernel_area = module.kernel_size[0] * module.kernel_size[1]
            total_macs += (
                out.numel()
                * (module.in_channels // module.groups)
                * kernel_area
            )
        elif isinstance(module, nn.Linear):
            total_macs += output.numel() * module.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(count_layer))
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, channels, height, height))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(previous_mode)
    return int(total_macs)


def final_checkpoint_metrics(server) -> dict:
    """Evaluate the saved global model once, after the last aggregation."""
    import torch
    from torch.utils.data import DataLoader
    from utils.data_utils import read_client_data

    model = copy.deepcopy(server.global_model).to("cpu")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for client_id in range(server.num_clients):
            dataset = read_client_data(server.dataset, client_id, is_train=False)
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            for features, labels in loader:
                logits = model(features)
                total_loss += loss_fn(logits, labels).item()
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_examples += int(labels.numel())
    return {
        "accuracy": total_correct / total_examples if total_examples else None,
        "cross_entropy": total_loss / total_examples if total_examples else None,
        "examples": total_examples,
    }


def batched_equivalent_evaluate(server) -> None:
    """Batch PFLlib's per-client evaluation without changing train inputs.

    It preserves pooled test accuracy and sample-weighted cross-entropy.
    AUC is auxiliary and is not among the MONZA paper's reported metrics.
    """
    import numpy as np
    import torch
    from utils.data_utils import read_data

    if not hasattr(server, "_repro_eval_data"):
        cache = []
        for client_id in range(server.num_clients):
            train = read_data(server.dataset, client_id, is_train=True)
            test = read_data(server.dataset, client_id, is_train=False)
            cache.append({
                "train_x": torch.as_tensor(train["x"], dtype=torch.float32),
                "train_y": torch.as_tensor(train["y"], dtype=torch.int64),
                "test_x": torch.as_tensor(test["x"], dtype=torch.float32),
                "test_y": torch.as_tensor(test["y"], dtype=torch.int64),
            })
        server._repro_eval_data = cache

    previous_mode = server.global_model.training
    server.global_model.eval()
    device = torch.device(server.device)
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    client_correct = []
    client_samples = []
    train_loss = 0.0
    train_samples = 0

    with torch.no_grad():
        for cached in server._repro_eval_data:
            images = cached["test_x"]
            labels = cached["test_y"]
            correct = 0
            for start in range(0, len(labels), 256):
                batch_x = images[start:start + 256].to(device)
                batch_y = labels[start:start + 256].to(device)
                logits = server.global_model(batch_x)
                correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            client_correct.append(correct / len(labels))
            client_samples.append(len(labels))

            train_images = cached["train_x"]
            train_labels = cached["train_y"]
            usable = (len(train_labels) // server.batch_size) * server.batch_size
            for start in range(0, usable, 256):
                batch_x = train_images[start:min(start + 256, usable)].to(device)
                batch_y = train_labels[start:min(start + 256, usable)].to(device)
                train_loss += loss_function(server.global_model(batch_x), batch_y).item()
            train_samples += usable

    # Evaluation's test/train DataLoader iterators each consume two global
    # PyTorch RNG draws in the original PFLlib implementation.
    for _ in range(4 * server.num_clients):
        torch.empty((), dtype=torch.int64).random_()
    for client in server.clients:
        client.model.eval()
    server.global_model.train(previous_mode)

    accuracy = sum(a * n for a, n in zip(client_correct, client_samples)) / sum(client_samples)
    cross_entropy = train_loss / train_samples
    server.rs_test_acc.append(accuracy)
    server.rs_test_auc.append(float("nan"))
    server.rs_train_loss.append(cross_entropy)
    print("Averaged Train Loss: {:.4f}".format(cross_entropy))
    print("Averaged Test Accuracy: {:.4f}".format(accuracy))
    print("Averaged Test AUC: not recomputed (not an article metric)")
    print("Std Test Accuracy: {:.4f}".format(float(np.std(client_correct))))


def save_run_record(
    server, method: str, run_dir: Path, wall_seconds: float,
    attack_config: dict | None = None,
) -> None:
    import h5py
    import numpy as np

    run_dir.mkdir(parents=True, exist_ok=True)
    accuracy = np.asarray(server.rs_test_acc, dtype=float)
    train_loss = np.asarray(server.rs_train_loss, dtype=float)
    round_seconds = np.asarray(server.Budget, dtype=float)
    with h5py.File(run_dir / "metrics.h5", "w") as h5:
        h5.create_dataset("rs_test_acc", data=accuracy)
        h5.create_dataset("rs_train_loss", data=train_loss)
        h5.create_dataset("rs_train_time", data=round_seconds)

    clients = server.clients
    per_client_seconds = []
    per_client_calls = []
    extra_label_flip_seconds = 0.0
    extra_label_flip_calls = 0
    processed_examples = 0
    for client in clients:
        train_calls = int(client.train_time_cost["num_rounds"])
        base_seconds = float(client.train_time_cost["total_cost"])
        extra_seconds = float(getattr(client, "_repro_label_flip_seconds", 0.0))
        extra_calls = int(getattr(client, "_repro_label_flip_calls", 0))
        if train_calls:
            per_client_seconds.append((base_seconds + extra_seconds) / train_calls)
            per_client_calls.append(train_calls)
        extra_label_flip_seconds += extra_seconds
        extra_label_flip_calls += extra_calls
        batches_examples = (client.train_samples // server.batch_size) * server.batch_size
        processed_examples += batches_examples * server.local_epochs * (train_calls + extra_calls)

    malicious_ids = {int(value) for value in server.index_malicious}
    metric_rounds = []
    if method == "fedmad":
        raw_rounds = server.detection_log
        if raw_rounds:
            first_ids = [int(value) for value in raw_rounds[0]["client_ids"]]
            first_truth = raw_rounds[0]["malicious_ground_truth"]
            malicious_ids = {
                client_id for client_id, is_malicious in zip(first_ids, first_truth)
                if int(is_malicious)
            }
        for row in raw_rounds:
            quarantined = {
                int(client_id)
                for client_id, duration in row.get("quarantine_after", {}).items()
                if int(duration) > 0
            }
            rates = confusion_rates(malicious_ids, quarantined, server.num_clients)
            metric_rounds.append({"round": int(row["round"]), **rates})
        (run_dir / "detection_log.json").write_text(
            json.dumps(raw_rounds, indent=2) + "\n", encoding="utf-8"
        )
    else:
        metric_rounds = list(getattr(server, "_repro_security_rounds", []))
        if not metric_rounds:
            for round_number in range(len(accuracy)):
                metric_rounds.append({
                    "round": round_number,
                    **confusion_rates(malicious_ids, set(), server.num_clients),
                })

    macs_per_example = model_forward_macs_per_example(
        server.global_model, server.dataset
    )
    estimated_train_flops = int(6 * macs_per_example * processed_examples)
    measured_client_seconds = (
        sum(per_client_seconds[index] * per_client_calls[index]
            for index in range(len(per_client_calls)))
    )
    compute_mflops_per_second = (
        estimated_train_flops / measured_client_seconds / 1_000_000
        if measured_client_seconds > 0 else None
    )

    final_metrics = final_checkpoint_metrics(server)
    final_model_path = Path("models") / server.dataset / f"{server.algorithm}_server.pt"
    if final_model_path.is_file():
        shutil.copy2(final_model_path, run_dir / "final_model.pt")

    attack_config = attack_config or {
        "profile": "repository",
        "noise_snr": 1.0,
        "label_flip_source": 0,
        "label_flip_target": 1,
    }
    if attack_config["profile"] == "article":
        attack_mix = {
            "selection": "one of the four attack types is selected uniformly for each malicious client and round; fixed by seed and paired across methods",
            "zero": "all local model parameters replaced by zeros",
            "random": (
                "additive Gaussian noise; per-tensor noise power equals signal power divided by linear SNR "
                f"({attack_config['noise_snr']}; table does not specify SNR)"
            ),
            "shuffle": "output channels/units independently permuted per Conv/Linear layer, with bias permuted in the same way",
            "label_flip": (
                f"targeted data label flip: class {attack_config['label_flip_source']} "
                f"to class {attack_config['label_flip_target']} (pair is an explicit default; table does not specify it)"
            ),
            "attack_start": "round 1 (the first loop iteration is clean with -ria 0)",
        }
    else:
        attack_mix = {
            "selection": "uniform deterministic choice keyed by seed, communication round, and client ID; assignments are stored below",
            "zero": "all model parameters set to zero (matches the article; implementation bug fixed)",
            "random": "repository implementation: parameter tensors replaced by Uniform[0,1] draws",
            "shuffle": "repository implementation: elements independently permuted in each parameter tensor",
            "label_flip": "repository implementation: cyclic shift y=(y+1) mod class_count",
            "attack_start": "round 1 (the first loop iteration is clean with -ria 0)",
        }

    record = {
        "method": method,
        "dataset": server.dataset,
        "seed": int(server.args.seed),
        "num_clients": int(server.num_clients),
        "malicious_clients": int(server.n_client_malicious),
        "malicious_client_ids": sorted(malicious_ids),
        "attack_profile": attack_config["profile"],
        "attack_parameters": {
            "random_noise_linear_snr": float(attack_config["noise_snr"]),
            "label_flip_source": int(attack_config["label_flip_source"]),
            "label_flip_target": int(attack_config["label_flip_target"]),
        },
        "attack_mix": attack_mix,
        "attack_schedule": planned_attack_schedule(
            int(server.args.seed), malicious_ids, int(server.global_rounds)
        ),
        "executed_attack_assignments": [
            {"round": round_number, "client_id": client_id, "attack": attack}
            for (round_number, client_id), attack in sorted(
                getattr(server, "_repro_attack_assignments", {}).items()
            )
        ],
        "rounds_argument": int(server.global_rounds) + 1,
        "pfl_global_rounds_argument": int(server.global_rounds),
        "recorded_evaluations": len(accuracy),
        "max_test_accuracy": float(accuracy.max()) if accuracy.size else None,
        "final_recorded_accuracy": float(accuracy[-1]) if accuracy.size else None,
        "final_recorded_train_loss": float(train_loss[-1]) if train_loss.size else None,
        "final_checkpoint_test": final_metrics,
        "total_round_wall_seconds": float(round_seconds.sum()),
        "mean_round_wall_seconds": float(round_seconds.mean()) if round_seconds.size else None,
        "average_client_train_seconds_per_call": (
            measured_client_seconds / sum(per_client_calls)
            if sum(per_client_calls) else None
        ),
        "total_client_train_seconds": float(measured_client_seconds),
        "client_train_calls": int(sum(per_client_calls)),
        "label_flip_retrain_calls": extra_label_flip_calls,
        "label_flip_retrain_seconds": float(extra_label_flip_seconds),
        "forward_macs_per_example": int(macs_per_example),
        "estimated_training_flops": estimated_train_flops,
        "estimated_training_mflops_per_second": compute_mflops_per_second,
        "wall_seconds_including_setup_and_final_evaluation": float(wall_seconds),
        "evaluation_mode": getattr(server, "_repro_evaluation_mode", "original per-client PFLlib evaluation"),
        "per_round": {
            "accuracy": accuracy.tolist(),
            "training_cross_entropy": train_loss.tolist(),
            "wall_seconds": round_seconds.tolist(),
            "fpr": [row["fpr"] for row in metric_rounds],
            "frr": [row["frr"] for row in metric_rounds],
        },
        "security_rounds": metric_rounds,
        "metric_notes": {
            "fpr_frr": "Fraction among all client IDs, using quarantine after each round as the malicious-client decision; 1.0 means 100%.",
            "training_flops": "Estimate: 2 FLOPs per multiply-accumulate and 3 forward-pass equivalents for forward/backward; uses actually trained examples and includes label-flip retraining.",
            "training_loss": "PFLlib's weighted cross-entropy over client training partitions, evaluated each communication loop.",
            "evaluation": getattr(server, "_repro_evaluation_mode", "original per-client PFLlib evaluation; AUC included"),
            "training_batch_order": "A per-client/per-round torch generator is used to avoid shared-RNG races among worker threads.",
            "attack_pairing": "Attack type and random tensor/permutation draws are deterministic per seed, communication round, and client ID; attack_schedule can be compared across methods.",
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REPRO_METRICS={run_dir / 'metrics.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fedmad", "monza", "fedavg", "fedavg-clean"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clients", type=int, default=100,
                        help="Number of clients in the selected partition")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=150,
                        help="Number of communication iterations to execute (exact count).")
    parser.add_argument("--malicious", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--attack-profile", choices=["repository", "article"], default="article",
        help="Use the repository attacks or the article-named Gaussian/channel/targeted variants.",
    )
    parser.add_argument(
        "--attack-noise-snr", type=float, default=1.0,
        help="Linear SNR for the article Gaussian-noise attack (default 1.0, equal signal/noise power).",
    )
    parser.add_argument("--label-flip-source", type=int, default=0)
    parser.add_argument("--label-flip-target", type=int, default=1)
    parser.add_argument("--exact-eval", action="store_true",
                        help="Use PFLlib's original per-client evaluation loops (slow on CPU).")
    args = parser.parse_args()
    if args.rounds < 1 or args.clients < 2:
        parser.error("--rounds must be positive and --clients must be at least 2")
    if args.malicious < 0 or args.malicious >= args.clients:
        parser.error("--malicious must be between 0 and clients - 1")
    num_classes = 100 if "Cifar100" in args.dataset else 10
    if args.attack_noise_snr <= 0:
        parser.error("--attack-noise-snr must be greater than zero")
    if not (0 <= args.label_flip_source < num_classes
            and 0 <= args.label_flip_target < num_classes):
        parser.error(f"label-flip classes must be in [0, {num_classes - 1}]")
    if args.label_flip_source == args.label_flip_target:
        parser.error("--label-flip-source and --label-flip-target must differ")
    run_dir = Path(os.environ["MONZA_REPRO_RUN_DIR"]).resolve()
    torch_threads = max(1, int(os.environ.get("MONZA_REPRO_TORCH_THREADS", "1")))
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    sys.path.insert(0, str(SYSTEM))
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but this PyTorch has no available CUDA device")
    torch.set_num_threads(torch_threads)
    from flcore.clients.clientmaliciousavg import ClientMaliciousAVG
    import flcore.clients.clientmaliciousavg as malicious_client_module
    from flcore.attack.attack import gaussian_noise_model, shuffle_layer_channels_model
    from flcore.clients.clientavg import clientAVG
    from flcore.clients.clientbase import Client
    from flcore.servers.serveravg import FedAvg
    from flcore.servers.serverbase import Server
    from flcore.servers.servermad import ServerMAD

    attack_assignments = {}

    def deterministic_send_local_model(client, round_number):
        round_number = int(round_number)
        if round_number <= int(client.round_init_atk):
            client.is_malicious = False
            return client.model

        attack_type = selected_attack(
            int(args.seed), round_number, int(client.id), float(client.rate_client_fake)
        )
        client.is_malicious = attack_type is not None
        if attack_type is None:
            return client.model
        if client.atack != "all":
            attack_type = str(client.atack)

        attack_assignments[(round_number, int(client.id))] = attack_type
        generator = torch.Generator(device="cpu")
        generator.manual_seed(attack_seed(int(args.seed), round_number, int(client.id)))
        if attack_type == "zero":
            return malicious_client_module.model_zeros(client.model, client.device)
        if attack_type == "random":
            if args.attack_profile == "article":
                return gaussian_noise_model(
                    client.model, snr=args.attack_noise_snr, generator=generator
                )
            return malicious_client_module.random_param(
                client.model, client.device, generator=generator
            )
        if attack_type == "shuffle":
            if args.attack_profile == "article":
                return shuffle_layer_channels_model(client.model, generator=generator)
            return malicious_client_module.shuffle_model(
                client.model, generator=generator
            )
        if attack_type == "label":
            if args.attack_profile == "article":
                return client._train_with_targeted_label_flip(
                    args.label_flip_source, args.label_flip_target
                )
            return client._train_with_label_flip()
        raise ValueError(f"Unknown deterministic attack: {attack_type}")

    ClientMaliciousAVG.send_local_model = deterministic_send_local_model

    start = time.time()
    original_send_models = Server.send_models

    def tagged_send_models(server):
        result = original_send_models(server)
        for client in server.clients:
            client._repro_round = server.current_round
        return result

    Server.send_models = tagged_send_models

    original_fedavg_init = FedAvg.__init__

    def isolated_fedavg_init(server, server_args, run_index):
        original_fedavg_init(server, server_args, run_index)
        server.csv_filename = str(run_dir / f"fpr_frr_cc{server.cc}.csv")
        if not Path(server.csv_filename).exists():
            with open(server.csv_filename, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["Round", "FPR", "FRR"])

    FedAvg.__init__ = isolated_fedavg_init

    original_load_train_data = Client.load_train_data

    def seeded_load_train_data(client, batch_size=None, is_malicious=False):
        loader = original_load_train_data(client, batch_size, is_malicious)
        if getattr(client, "_repro_training_active", False):
            pass_number = int(getattr(client, "_repro_training_pass", 0))
            local_seed = (
                int(args.seed) * 1_000_003
                + (int(getattr(client, "_repro_round", 0)) + 1) * 100_003
                + int(client.id) * 1_009
                + pass_number
            ) % (2**63 - 1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(local_seed)
            loader.generator = generator
            if hasattr(loader.sampler, "generator"):
                loader.sampler.generator = generator
        return loader

    Client.load_train_data = seeded_load_train_data

    original_client_train = clientAVG.train

    def seeded_client_train(client):
        client._repro_training_active = True
        client._repro_training_pass = 0
        try:
            return original_client_train(client)
        finally:
            client._repro_training_active = False

    clientAVG.train = seeded_client_train

    original_flip = ClientMaliciousAVG._train_with_label_flip
    original_target_flip = ClientMaliciousAVG._train_with_targeted_label_flip

    def timed_label_flip(client):
        begin = time.time()
        client._repro_training_active = True
        client._repro_training_pass = 1
        try:
            return original_flip(client)
        finally:
            client._repro_training_active = False
            client._repro_label_flip_calls = getattr(client, "_repro_label_flip_calls", 0) + 1
            client._repro_label_flip_seconds = (
                getattr(client, "_repro_label_flip_seconds", 0.0) + time.time() - begin
            )

    ClientMaliciousAVG._train_with_label_flip = timed_label_flip

    def timed_target_label_flip(client, source_label, target_label):
        begin = time.time()
        client._repro_training_active = True
        client._repro_training_pass = 1
        try:
            return original_target_flip(client, source_label, target_label)
        finally:
            client._repro_training_active = False
            client._repro_label_flip_calls = getattr(client, "_repro_label_flip_calls", 0) + 1
            client._repro_label_flip_seconds = (
                getattr(client, "_repro_label_flip_seconds", 0.0) + time.time() - begin
            )

    ClientMaliciousAVG._train_with_targeted_label_flip = timed_target_label_flip

    original_avg_train = FedAvg.train
    original_compute = FedAvg.compute_fpr_frr

    def wrapped_avg_train(server):
        server._repro_security_rounds = []

        def tracked_compute():
            fpr, frr = original_compute(server)
            if server.cc == 3:
                blocked = {
                    int(client_id)
                    for client_id, status in server.client_quarantine_dict.items()
                    if status["roundsQuarent"] > 0
                }
                rates = confusion_rates(
                    {int(value) for value in server.index_malicious},
                    blocked,
                    server.num_clients,
                )
                server._repro_security_rounds.append({
                    "round": int(server.current_round),
                    **rates,
                    "fpr_as_printed_by_repository": float(fpr),
                    "frr_as_printed_by_repository": float(frr),
                })
            return fpr, frr

        server.compute_fpr_frr = tracked_compute
        if not args.exact_eval:
            server.evaluate = lambda *unused_args, **unused_kwargs: batched_equivalent_evaluate(server)
            server._repro_evaluation_mode = "batched equivalent for accuracy/loss; auxiliary AUC omitted"
        original_avg_train(server)
        if not server._repro_security_rounds:
            for round_number in range(len(server.rs_test_acc)):
                server._repro_security_rounds.append({
                    "round": round_number,
                    **confusion_rates(
                        {int(value) for value in server.index_malicious},
                        set(),
                        server.num_clients,
                    ),
                })
        server._repro_attack_assignments = attack_assignments
        save_run_record(
            server, args.method, run_dir, time.time() - start,
            attack_config={
                "profile": args.attack_profile,
                "noise_snr": args.attack_noise_snr,
                "label_flip_source": args.label_flip_source,
                "label_flip_target": args.label_flip_target,
            },
        )

    original_mad_train = ServerMAD.train

    def wrapped_mad_train(server):
        if not args.exact_eval:
            server.evaluate = lambda *unused_args, **unused_kwargs: batched_equivalent_evaluate(server)
            server._repro_evaluation_mode = "batched equivalent for accuracy/loss; auxiliary AUC omitted"
        original_mad_train(server)
        server._repro_attack_assignments = attack_assignments
        save_run_record(
            server, args.method, run_dir, time.time() - start,
            attack_config={
                "profile": args.attack_profile,
                "noise_snr": args.attack_noise_snr,
                "label_flip_source": args.label_flip_source,
                "label_flip_target": args.label_flip_target,
            },
        )

    FedAvg.train = wrapped_avg_train
    ServerMAD.train = wrapped_mad_train

    if args.method == "fedmad":
        algorithm, cc = "MAD", 5
        malicious_count = args.malicious
    elif args.method == "monza":
        algorithm, cc = "FedAvg", 3
        malicious_count = args.malicious
    else:
        algorithm, cc = "FedAvg", 5
        malicious_count = 0 if args.method == "fedavg-clean" else args.malicious

    goal = f"monza_article_{args.dataset}_{args.clients}c_{args.method}_seed{args.seed}"
    sys.argv = [
        str(SYSTEM / "main.py"),
        "-go", goal,
        "-dev", args.device,
        "-data", args.dataset,
        "-ncl", str(num_classes),
        "-m", "CNN",
        "-lbs", "10",
        "-lr", "0.005",
        "-ld", "True",
        "-ldg", "0.99",
        # PFLlib loops with range(global_rounds + 1), so subtract one to
        # make this CLI option represent the exact number of iterations.
        "-gr", str(args.rounds - 1),
        "-ls", "1",
        "-algo", algorithm,
        "-jr", "1.0",
        "-nc", str(args.clients),
        "-t", "1",
        "-eg", "1",
        "-nmc", str(malicious_count),
        "-atk", "all",
        "-rfake", "1",
        "-ria", "0",
        "-cc", str(cc),
        "--seed", str(args.seed),
    ]
    runpy.run_path(str(SYSTEM / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
