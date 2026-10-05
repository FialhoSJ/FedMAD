#!/usr/bin/env python3
"""Launch reproducible, paired-seed experiments using the MONZA article setup."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASETS = ["MONZA_MNIST_A02", "MONZA_Cifar10_A02", "MONZA_Cifar100_A02"]
METHODS = ["fedmad", "monza"]


def cpu_model_name() -> str:
    candidate = Path("/proc/cpuinfo")
    if candidate.is_file():
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS),
                        help="Comma-separated dataset names, or 'all'.")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="Comma-separated methods: fedmad,monza; fedavg and fedavg-clean are optional.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated paired seeds (paper reports 10 executions).")
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--malicious", type=int, default=30)
    parser.add_argument(
        "--attack-profile", choices=["repository", "article"], default="article",
        help="Use repository attack implementations or the article-named variants.",
    )
    parser.add_argument("--attack-noise-snr", type=float, default=1.0,
                        help="Linear SNR for article Gaussian noise; default 1.0 (0 dB).")
    parser.add_argument("--label-flip-source", type=int, default=0)
    parser.add_argument("--label-flip-target", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Execution device; auto selects CUDA when available.")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="Intra-op CPU threads per client; 1 limits oversubscription on CPU runs.")
    parser.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"),
                        help="Python interpreter; defaults to this repository's virtual environment.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have metrics.json.")
    parser.add_argument("--exact-eval", action="store_true",
                        help="Use the original slow per-client PFLlib evaluation loop.")
    args = parser.parse_args()

    dataset_names = DATASETS if args.datasets.lower() == "all" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.rounds < 1 or args.malicious < 0 or args.torch_threads < 1:
        parser.error("rounds and torch threads must be positive; malicious must be non-negative")
    if args.attack_noise_snr <= 0:
        parser.error("--attack-noise-snr must be greater than zero")
    if args.label_flip_source == args.label_flip_target:
        parser.error("--label-flip-source and --label-flip-target must differ")
    if set(methods) - {"fedmad", "fedavg", "monza", "fedavg-clean"}:
        parser.error("unknown method in --methods")
    for name in dataset_names:
        if name not in DATASETS:
            parser.error(f"unknown dataset {name!r}; choices: {', '.join(DATASETS)}")
        class_count = 100 if "Cifar100" in name else 10
        if not (0 <= args.label_flip_source < class_count
                and 0 <= args.label_flip_target < class_count):
            parser.error(f"label-flip classes for {name} must be in [0, {class_count - 1}]")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but this PyTorch has no available CUDA device")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    evaluation_suffix = "exact_eval_seeded" if args.exact_eval else "batched_eval_seeded"
    attack_suffix = "_article_attacks" if args.attack_profile == "article" else ""
    output_root = REPO / "results" / "monza_article_replication" / f"rounds_{args.rounds}{attack_suffix}_{evaluation_suffix}_{device}"
    output_root.mkdir(parents=True, exist_ok=True)
    environment = {
        "protocol": "MONZA article matched comparison",
        "rounds_argument": args.rounds,
        "loop_updates_in_pfl_lib": args.rounds,
        "pfl_global_rounds_argument": args.rounds - 1,
        "malicious_clients": args.malicious,
        "num_clients": 100,
        "dirichlet_alpha": 0.2,
        "attack_profile": args.attack_profile,
        "attack_parameters": {
            "random_noise_linear_snr": args.attack_noise_snr,
            "label_flip_source": args.label_flip_source,
            "label_flip_target": args.label_flip_target,
            "note": "The article summary table does not state these three attack-specific values; profile defaults are recorded per run.",
        },
        "monza_implementation": "repository score-based branch -cc 3; the paper table's 2-cluster setting is not a parameter of this branch",
        "methods": methods,
        "datasets": dataset_names,
        "seeds": seeds,
        "device": device,
        "torch_threads": args.torch_threads,
        "evaluation_mode": "original per-client PFLlib evaluation" if args.exact_eval else "batched evaluation; preserves accuracy/loss and skips auxiliary AUC",
        "training_batch_order": "fixed per client and round",
        "attack_pairing": "same deterministic attack type and random draw keyed by seed, round, and client ID across methods; schedule saved in metrics.json",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
        "processor": cpu_model_name(),
        "logical_cpu_count": os.cpu_count(),
        "paper_hardware": "2 x NVIDIA RTX 4090; Intel Core i9-13900K; 128 GB RAM",
    }
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    # Preserve a venv's `bin/python` symlink: resolving it points at the base
    # interpreter and drops the virtual environment's site-packages.
    python = str(Path(args.python).expanduser().absolute())
    if not Path(python).is_file():
        parser.error(f"Python interpreter does not exist: {python}")

    total = len(dataset_names) * len(methods) * len(seeds)
    completed = 0
    for dataset in dataset_names:
        for seed in seeds:
            for method in methods:
                run_dir = output_root / dataset / method / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                if args.resume and metrics_path.is_file():
                    completed += 1
                    print(f"[{completed}/{total}] skip existing {dataset} {method} seed={seed}", flush=True)
                    continue

                command = [
                    python, "-u", str(HERE / "run_single.py"),
                    "--method", method,
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--rounds", str(args.rounds),
                    "--malicious", str(args.malicious),
                    "--device", device,
                    "--attack-profile", args.attack_profile,
                    "--attack-noise-snr", str(args.attack_noise_snr),
                    "--label-flip-source", str(args.label_flip_source),
                    "--label-flip-target", str(args.label_flip_target),
                ]
                if args.exact_eval:
                    command.append("--exact-eval")
                env = os.environ.copy()
                env["MONZA_REPRO_RUN_DIR"] = str(run_dir)
                env["MONZA_REPRO_TORCH_THREADS"] = str(args.torch_threads)
                env["OMP_NUM_THREADS"] = str(args.torch_threads)
                env["MKL_NUM_THREADS"] = str(args.torch_threads)
                env.setdefault("MPLCONFIGDIR", "/tmp/mpl-monza-article")
                started = time.time()
                print(f"[{completed + 1}/{total}] start {dataset} {method} seed={seed}", flush=True)
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO / "system",
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        log.write(line)
                        round_match = re.search(r"Round number:\s*(\d+)", line)
                        if round_match:
                            print(
                                f"    round {int(round_match.group(1)) + 1}/{args.rounds}",
                                flush=True,
                            )
                        elif " time cost " in line:
                            print(f"    {line.strip()}", flush=True)
                    result_code = process.wait()
                elapsed = time.time() - started
                if result_code != 0 or not metrics_path.is_file():
                    tail = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                    print("\n".join(tail), file=sys.stderr)
                    raise SystemExit(
                        f"Run failed ({result_code}): {dataset} {method} seed={seed}; "
                        f"see {run_dir / 'run.log'}"
                    )
                completed += 1
                print(
                    f"[{completed}/{total}] done {dataset} {method} seed={seed} "
                    f"in {elapsed / 60:.1f} min",
                    flush=True,
                )

    print(f"All requested runs finished. Results: {output_root}", flush=True)
    print(
        f"Generate article-style comparisons with: {python} {HERE / 'analyze_results.py'} "
        f"--results {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
