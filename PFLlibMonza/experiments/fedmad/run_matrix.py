"""Run a declared FedMAD comparison grid with separate logs per seed."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ABLATIONS = {
    "history": "--mad_ablate_history",
    "reputation": "--mad_ablate_reputation",
    "temporal": "--mad_ablate_temporal",
    "meta": "--mad_ablate_meta",
    "validator": "--mad_ablate_validator",
}


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    grid_path = args.grid.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    base_config = (grid_path.parent / grid["base_config"]).resolve()
    base = json.loads(base_config.read_text(encoding="utf-8"))
    config_sha256 = hashlib.sha256(base_config.read_bytes()).hexdigest()
    project = Path(__file__).resolve().parents[2]
    system = project / "system"
    output = (project / "results" / safe_name(grid.get("output", "fedmad_redesign"))).resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    repo = project.parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    completed = set()
    if args.resume and manifest.exists():
        latest = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            latest[row["run_id"]] = row
        for row in latest.values():
            if (row["status"] == "complete"
                    and row.get("config_sha256") == config_sha256
                    and row.get("final_evaluation")
                    and Path(row["final_evaluation"]).exists()):
                completed.add(row["run_id"])
    for attack in grid["attacks"]:
        for method in grid["methods"]:
            algorithm = method["algorithm"]
            ablation = method.get("ablation", "full")
            if ablation not in ("full", *ABLATIONS):
                raise ValueError(f"unknown ablation: {ablation}")
            for seed in grid["seeds"]:
                run_id = safe_name(f"{base['dataset']}_{attack}_{method['name']}_seed{seed}")
                if run_id in completed:
                    continue
                command = [
                    sys.executable, str(system / "main.py"), "--config", str(base_config),
                    "-algo", algorithm, "-atk", attack, "--seed", str(seed),
                    "-go", run_id, "--mad_run_id", run_id,
                ]
                if ablation != "full":
                    command.append(ABLATIONS[ablation])
                if method.get("fixed_defense"):
                    command.extend(["-mad_fixed_defense", method["fixed_defense"]])
                if args.dry_run:
                    print(" ".join(command))
                    continue
                stdout_path = output / f"{run_id}.txt"
                with stdout_path.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=system, stdout=stream, stderr=subprocess.STDOUT, check=False)
                detection_path = project / "results" / (
                    f"detection_log_{base['dataset']}_{algorithm}_cc{base.get('cluster_comparation', 0)}_{run_id}.json"
                )
                record = {
                    "run_id": run_id,
                    "dataset": base["dataset"], "attack": attack,
                    "method": method["name"], "algorithm": algorithm,
                    "ablation": ablation, "seed": seed,
                    "status": "complete" if result.returncode == 0 else "failed",
                    "exit_code": result.returncode,
                    "stdout": str(stdout_path),
                    "detection_log": str(detection_path) if detection_path.exists() else None,
                    "final_evaluation": str(project / "results" / f"final_eval_{run_id}.json"),
                    "config": str(base_config),
                    "config_sha256": config_sha256,
                    "git_revision": revision,
                    "git_dirty": dirty,
                }
                with manifest.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(run_id, record["status"], flush=True)


if __name__ == "__main__":
    main()
