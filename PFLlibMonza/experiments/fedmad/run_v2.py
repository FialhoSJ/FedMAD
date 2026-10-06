"""Phased V2 experiments with paired data/seeds, code snapshots and exact jobs."""

import argparse
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_v2_datasets import prepare_dataset, prepared_name

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "system"))
from flcore.madsystem.config import load_experiment_config

ABLATIONS = {"history": "mad_ablate_history", "reputation": "mad_ablate_reputation", "population": "mad_ablate_population",
             "historical_deviation": "mad_ablate_temporal", "meta": "mad_ablate_meta", "validator": "mad_ablate_validator"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(output, write=True):
    files = list((PROJECT / "system").rglob("*.py")) + list((PROJECT / "experiments").rglob("*.py")) + [PROJECT / "dataset/utils/dataset_utils.py"]
    hashes = {}
    for path in sorted(files):
        relative = path.relative_to(PROJECT)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    code_hash = digest(hashes)
    snapshot_dir = output / "code_snapshots" / code_hash
    if write:
        for path in sorted(files):
            target = snapshot_dir / path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (snapshot_dir / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        import numpy
        import torch
        environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                       "numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                       "deterministic": "one torch thread; private RNG per seed/client/round; paired client sampling before quarantine"}
        (snapshot_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return code_hash


def planned_jobs(grid, base):
    for distribution, attack, method, seed, alpha in itertools.product(grid.get("distributions", ["iid"]), grid["attacks"], grid["methods"], grid["seeds"], grid.get("history_alphas", [base.get("mad_history_alpha", 0.9)])):
        config = dict(base)
        config.update(grid.get("overrides", {}))
        config["mad_history_alpha"] = alpha
        config["seed"] = int(seed)
        config["mad_version"] = "v2"
        config["mad_deterministic"] = True
        config["global_rounds"] = int(grid.get("rounds", config.get("global_rounds", 149) + 1)) - 1
        config["num_clients"] = int(grid.get("clients", config.get("num_clients", 20)))
        config["n_client_malicious"] = int(config["num_clients"] * float(grid.get("malicious_fraction", 0.2)))
        for name in ABLATIONS.values():
            config[name] = False
        ablation = method.get("ablation")
        if ablation:
            if ablation not in ABLATIONS:
                raise ValueError(f"unknown V2 ablation: {ablation}")
            config[ABLATIONS[ablation]] = True
        attack = {"type": attack} if isinstance(attack, str) else dict(attack)
        config["atack"] = attack["type"]
        config["round_init_atk"] = int(attack.get("start_round", grid.get("attack_start_round", 30))) - 1
        if config["atack"] == "clean":
            config["n_client_malicious"] = 0
            config["atack"] = "sign_flipping"
        config["attack_source_label"] = attack.get("source_label")
        config["attack_target_label"] = attack.get("target_label")
        for key, argument in (("scale", "attack_scale"), ("noise_snr", "attack_noise_snr"), ("rate", "rate_client_fake")):
            if key in attack:
                config[argument] = attack[key]
        config["algorithm"] = method.get("algorithm", "MAD")
        config["cluster_comparation"] = 3 if method["name"] == "monza" else 5
        if method.get("fixed_defense"):
            config["mad_fixed_defense"] = method["fixed_defense"]
        config["mad_data_distribution"] = {"type": "iid" if distribution == "iid" else "dirichlet", "alpha": None if distribution == "iid" else float(distribution), "seed": seed}
        yield distribution, attack, method, seed, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    grid_path = args.grid.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    base = load_experiment_config(grid_path.parent / grid["base_config"])
    source = base["dataset"]
    output = PROJECT / "results" / re.sub(r"[^A-Za-z0-9_-]", "_", grid.get("output", "fedmad_v2"))
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    code_hash = snapshot(output, write=not args.dry_run)
    manifest_path = output / "manifest.jsonl"
    latest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            latest[row["run_id"]] = row
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT.parent, capture_output=True, text=True, check=True).stdout.strip()
    completed = failures = 0
    for distribution, attack, method, seed, config in planned_jobs(grid, base):
        maximum = grid.get("max_examples")
        if args.dry_run:
            dataset = prepared_name(source, config["num_clients"], distribution, seed, maximum)
            validation = PROJECT / "dataset" / dataset / "validation.npz"
        else:
            dataset, validation = prepare_dataset(source, config["num_clients"], distribution, seed, maximum,
                                                   grid.get("validation_fraction", 0.1), config.get("batch_size", 10))
        config["dataset"] = dataset
        config["mad_validation_data"] = str(validation)
        phase = attack.get("phase", "evaluation")
        config_hash = digest({"config": config, "code_sha256": code_hash, "phase": phase})
        protocol = {key: value for key, value in config.items() if key not in ("seed", "dataset", "mad_validation_data", "mad_data_distribution")}
        protocol_hash = digest({"config": protocol, "source_dataset": source, "distribution": distribution,
                                "phase": phase, "attack": attack, "method": method, "max_examples": maximum,
                                "validation_fraction": grid.get("validation_fraction", 0.1)})
        run_id = re.sub(r"[^A-Za-z0-9_-]", "_", f"{dataset}_{attack['type']}_{method['name']}_ha{config['mad_history_alpha']}_{config_hash[:10]}")[:80]
        # Include the hash at the beginning to avoid suffix loss on long names.
        run_id = config_hash[:10] + "_" + run_id[:69]
        config["mad_run_id"] = run_id
        config["goal"] = run_id
        final_path = PROJECT / "results" / f"final_eval_{run_id}.json"
        detection = PROJECT / "results" / f"detection_log_{dataset}_{config['algorithm']}_cc{config['cluster_comparation']}_{run_id}.json"
        previous = latest.get(run_id, {})
        if args.resume and previous.get("status") == "complete" and final_path.is_file() and detection.is_file():
            print(f"skip completed {run_id}", flush=True)
            continue
        if args.dry_run:
            print(json.dumps({"method": method["name"], "attack": attack["type"], "phase": attack.get("phase", "evaluation"), "seed": seed,
                              "distribution": distribution, "rounds": config["global_rounds"] + 1, "history_alpha": config["mad_history_alpha"], "dataset": dataset}))
            continue
        config_path = output / f"{run_id}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        log = output / f"{run_id}.log"
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        print(f"start {method['name']} {attack['type']} {distribution} seed={seed}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, str(PROJECT / "system/main.py"), "--config", str(config_path)],
                                    cwd=PROJECT / "system", env=environment, stdout=stream, stderr=subprocess.STDOUT)
        status = "complete" if result.returncode == 0 and final_path.is_file() and detection.is_file() else "failed"
        record = {"run_id": run_id, "dataset": dataset, "source_dataset": source, "distribution": distribution,
                  "attack": attack["type"], "phase": attack.get("phase", "evaluation"), "method": method["name"], "algorithm": config["algorithm"],
                  "seed": seed, "history_alpha": config["mad_history_alpha"], "status": status, "exit_code": result.returncode,
                  "config": str(config_path), "config_sha256": config_hash, "protocol_sha256": protocol_hash,
                  "code_sha256": code_hash, "git_revision": revision,
                  "code_snapshot": str(output / "code_snapshots" / code_hash), "stdout": str(log), "detection_log": str(detection), "final_evaluation": str(final_path)}
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        completed += status == "complete"
        failures += status == "failed"
        print(f"{status}: {run_id}", flush=True)
        if status == "failed":
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-18:]), file=sys.stderr)
    if not args.dry_run:
        print(f"Finished: {completed} completed, {failures} failed. Manifest: {manifest_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
