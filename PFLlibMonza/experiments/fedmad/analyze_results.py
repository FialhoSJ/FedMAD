"""Summarize completed FedMAD matrix runs by independent seed."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import h5py


DIMENSIONS = ("dataset", "distribution", "attack", "method", "phase", "history_alpha", "code_sha256", "protocol_sha256")


def comparison_key(record):
    """Only pool independent seeds with identical code and scientific protocol."""
    return (record.get("source_dataset", record["dataset"]), str(record.get("distribution", "existing")),
            record["attack"], record["method"], record.get("phase", "evaluation"),
            str(record.get("history_alpha", "")), record.get("code_sha256", "legacy"), record.get("protocol_sha256", "legacy"))


def metric_statistics(runs, column):
    values = [run[column] for run in runs if run.get(column) is not None]
    return {"mean": mean(values) if values else "", "sd": stdev(values) if len(values) > 1 else "", "n": len(values)}


def summarize_log(path):
    if not path or not Path(path).exists():
        return {}
    rounds = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = {key: sum(int(row.get("detection", {}).get(key, 0)) for row in rounds) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    eligible = [row for row in rounds if row.get("client_ids")]
    metrics = {
        "detection_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tpr": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "defense_activation_rate": sum(bool(row.get("robust_defense_activated")) for row in eligible) / len(eligible) if eligible else 0.0,
        "round_seconds": mean(float(row["round_seconds"]) for row in rounds if "round_seconds" in row) if any("round_seconds" in row for row in rounds) else None,
    }
    metrics.update({"precision": precision, "fnr": fn / (fn + tp) if fn + tp else 0.0})
    summary = rounds[-1].get("evaluation_summary", {}) if rounds else {}
    for key in ("unnecessary_defense_rate", "detection_latency", "latency_censored_fraction"):
        metrics[key] = summary.get(key)
    peaks = [row["peak_memory_bytes"] for row in rounds if row.get("peak_memory_bytes") is not None]
    metrics["peak_memory_bytes"] = max(peaks) if peaks else None
    seconds = sum(float(row.get("round_seconds", 0)) for row in rounds)
    sentinel = sum(float(row.get("sentinel_seconds", 0)) for row in rounds)
    metrics["sentinel_overhead"] = sentinel / seconds if seconds else None
    metrics["sentinel_seconds_per_round"] = sentinel / len(rounds) if rounds else None
    metrics["memory_risk_seconds_per_round"] = sum(float(row.get("memory_risk_seconds", 0)) for row in rounds) / len(rounds) if rounds else None
    metrics["meta_seconds_per_round"] = sum(float(row.get("meta_seconds", 0)) for row in rounds) / len(rounds) if rounds else None
    metrics["validator_seconds_per_round"] = sum(float(attempt.get("validation_seconds", 0)) for row in rounds for attempt in row.get("defense_attempts", [])) / len(rounds) if rounds else None
    risk_counts = {key: sum(row.get("risk_detection", {}).get(key, 0) for row in rounds) for key in ("tp", "fp", "fn", "tn")}
    if any("risk_detection" in row for row in rounds):
        tp2, fp2, fn2, tn2 = (risk_counts[key] for key in ("tp", "fp", "fn", "tn"))
        metrics.update({"risk_tpr": tp2 / (tp2 + fn2) if tp2 + fn2 else 0.0,
                        "risk_fpr": fp2 / (fp2 + tn2) if fp2 + tn2 else 0.0,
                        "risk_detection_f1": 2 * tp2 / (2 * tp2 + fp2 + fn2) if 2 * tp2 + fp2 + fn2 else 0.0})
    return metrics


def h5_results(project, record):
    pattern = f"{record['dataset']}_{record['algorithm']}_*_{record['run_id']}_0.h5"
    matches = list((project / "results").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one HDF5 result for {record['run_id']}, found {len(matches)}")
    with h5py.File(matches[0], "r") as result:
        accuracy = result["rs_test_acc"][:]
        loss = result["rs_train_loss"][:]
    metrics = {"last_recorded_accuracy": float(accuracy[-1])}
    if "code_sha256" not in record:
        metrics["last_pre_update_accuracy"] = float(accuracy[-1])
    final_evaluation = record.get("final_evaluation")
    if not final_evaluation or not Path(final_evaluation).exists():
        raise ValueError(f"missing post-update evaluation for {record['run_id']}")
    final = json.loads(Path(final_evaluation).read_text(encoding="utf-8"))
    metrics.update({"final_accuracy": float(final["accuracy"]), "final_train_loss": float(final["train_loss"])})
    for key in ("clean_accuracy", "robust_accuracy", "attack_success_rate"):
        metrics[key] = final.get(key)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    project = Path(__file__).resolve().parents[2]
    groups = defaultdict(list)
    grouped_seeds = defaultdict(set)
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
        key = comparison_key(record)
        if record["seed"] in grouped_seeds[key]:
            raise ValueError(f"duplicate seed {record['seed']} in one comparison protocol: {key}; choose independent runs")
        grouped_seeds[key].add(record["seed"])
        groups[key].append(metrics)
    columns = ("final_accuracy", "last_recorded_accuracy", "last_pre_update_accuracy", "final_train_loss", "clean_accuracy", "robust_accuracy", "attack_success_rate",
               "detection_f1", "tpr", "fpr", "fnr", "precision", "risk_tpr", "risk_fpr", "risk_detection_f1",
               "detection_latency", "latency_censored_fraction", "defense_activation_rate", "unnecessary_defense_rate", "round_seconds",
               "peak_memory_bytes", "sentinel_overhead", "sentinel_seconds_per_round", "memory_risk_seconds_per_round", "meta_seconds_per_round", "validator_seconds_per_round")
    output = manifest.parent / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        dimensions = list(DIMENSIONS)
        writer = csv.DictWriter(stream, fieldnames=dimensions + ["seeds"] + [f"{col}_{stat}" for col in columns for stat in ("mean", "sd", "n")])
        writer.writeheader()
        for key, runs in sorted(groups.items()):
            row = dict(zip(dimensions, key))
            row["seeds"] = len(runs)
            for column in columns:
                for statistic, value in metric_statistics(runs, column).items():
                    row[f"{column}_{statistic}"] = value
            writer.writerow(row)
    (manifest.parent / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
