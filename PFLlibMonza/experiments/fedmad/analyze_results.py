"""Summarize completed FedMAD matrix runs by independent seed."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import h5py


def summarize_log(path):
    if not path or not Path(path).exists():
        return {}
    rounds = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = {key: sum(int(row.get("detection", {}).get(key, 0)) for row in rounds) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    eligible = [row for row in rounds if row.get("client_ids")]
    return {
        "detection_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tpr": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "defense_activation_rate": sum(bool(row.get("robust_defense_activated")) for row in eligible) / len(eligible) if eligible else 0.0,
        "round_seconds": mean(float(row["round_seconds"]) for row in rounds if "round_seconds" in row) if any("round_seconds" in row for row in rounds) else None,
    }


def h5_results(project, record):
    pattern = f"{record['dataset']}_{record['algorithm']}_*_{record['run_id']}_0.h5"
    matches = list((project / "results").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one HDF5 result for {record['run_id']}, found {len(matches)}")
    with h5py.File(matches[0], "r") as result:
        accuracy = result["rs_test_acc"][:]
        loss = result["rs_train_loss"][:]
    metrics = {"last_pre_update_accuracy": float(accuracy[-1])}
    final_evaluation = record.get("final_evaluation")
    if not final_evaluation or not Path(final_evaluation).exists():
        raise ValueError(f"missing post-update evaluation for {record['run_id']}")
    final = json.loads(Path(final_evaluation).read_text(encoding="utf-8"))
    metrics.update({"final_accuracy": float(final["accuracy"]), "final_train_loss": float(final["train_loss"])})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    project = Path(__file__).resolve().parents[2]
    groups = defaultdict(list)
    failures = []
    latest = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        latest[record["run_id"]] = record
    for record in latest.values():
        if record["status"] != "complete":
            failures.append(record)
            continue
        metrics = h5_results(project, record)
        metrics.update(summarize_log(record.get("detection_log")))
        groups[(record["dataset"], record["attack"], record["method"])].append(metrics)
    columns = ("final_accuracy", "last_pre_update_accuracy", "final_train_loss", "detection_f1", "tpr", "fpr", "defense_activation_rate", "round_seconds")
    output = manifest.parent / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "attack", "method", "seeds"] + [f"{col}_{stat}" for col in columns for stat in ("mean", "sd")])
        writer.writeheader()
        for key, runs in sorted(groups.items()):
            row = dict(zip(("dataset", "attack", "method"), key))
            row["seeds"] = len(runs)
            for column in columns:
                values = [run[column] for run in runs if run.get(column) is not None]
                row[f"{column}_mean"] = mean(values) if values else ""
                row[f"{column}_sd"] = stdev(values) if len(values) > 1 else ""
            writer.writerow(row)
    (manifest.parent / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
