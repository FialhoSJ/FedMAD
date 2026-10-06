"""Plot paired FedMAD comparison runs listed in a run manifest."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from analyze_results import summarize_log


def read_run(project: Path, row: dict):
    if "code_sha256" in row:
        rounds = json.loads(Path(row["detection_log"]).read_text(encoding="utf-8"))
        evaluated = [entry for entry in rounds if entry.get("global_metrics")]
        x = np.asarray([entry["round"] for entry in evaluated])
        curve = np.asarray([entry["global_metrics"]["accuracy"] for entry in evaluated], dtype=float)
    else:
        x = None
        curve = None
    pattern = f"{row['dataset']}_{row['algorithm']}_*_{row['run_id']}_0.h5"
    candidates = list((project / "results").glob(pattern))
    if len(candidates) != 1:
        raise ValueError(f"{row['run_id']}: expected one HDF5 file, found {len(candidates)}")
    if curve is None:
        with h5py.File(candidates[0], "r") as h5:
            curve = np.asarray(h5["rs_test_acc"], dtype=float)
        x = np.arange(len(curve))
    final_path = Path(row["final_evaluation"])
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {}
    return x, curve, final.get("accuracy", curve[-1] if len(curve) else np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    project = Path(__file__).resolve().parents[2]
    output = manifest.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    latest = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[row["run_id"]] = row
    groups = {}
    for row in latest.values():
        if row["status"] != "complete":
            continue
        context = (row.get("source_dataset", row["dataset"]), str(row.get("distribution", "existing")),
                   row.get("phase", "evaluation"), str(row.get("history_alpha", "")), row.get("code_sha256", "legacy"))
        key = (context, row["attack"], row["method"], row.get("protocol_sha256", "legacy"))
        x, curve, final_accuracy = read_run(project, row)
        bucket = groups.setdefault(key, [])
        if any(run[0] == int(row["seed"]) for run in bucket):
            raise ValueError(f"duplicate seed in chart comparison: {key}")
        bucket.append((int(row["seed"]), x, curve, float(final_accuracy), summarize_log(row.get("detection_log"))))
    if not groups:
        raise SystemExit("Manifest has no completed runs")

    contexts = sorted({(key[0], key[1]) for key in groups})
    for context, attack in contexts:
        selected = {(method, protocol): runs for (group_context, group_attack, method, protocol), runs in groups.items()
                    if group_context == context and group_attack == attack}
        target = output
        if context[-1] != "legacy":
            name = "_".join(context[:-1]) + "_" + context[-1][:10]
            target = output / name.replace(".", "p")
            target.mkdir(parents=True, exist_ok=True)
        title_suffix = f"{attack}; {context[0]} / {context[1]}; {context[2]}"
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        labels = []
        for (method, protocol), runs in sorted(selected.items()):
            label = method if sum(key[0] == method for key in selected) == 1 else f"{method} [{protocol[:6]}]"
            labels.append(label)
            x = runs[0][1]
            if not all(np.array_equal(x, run[1]) for run in runs):
                raise ValueError("evaluation rounds differ between seeds; cannot silently truncate curves")
            values = np.asarray([run[2] for run in runs])
            length = len(x)
            avg = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(length)
            ax.plot(x, avg, marker="o", markersize=3, label=f"{label} (n={len(runs)})")
            if len(runs) > 1:
                ax.fill_between(x, avg - spread, avg + spread, alpha=0.16)
        ax.set(title=f"Test accuracy — {title_suffix}", xlabel="Round (zero based)" if context[-1] != "legacy" else "Evaluation index", ylabel="Test accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(target / f"accuracy_by_round_{attack}.png", dpi=180)
        fig.savefig(target / f"accuracy_by_round_{attack}.pdf")
        plt.close(fig)

        means, errors = [], []
        for runs in [selected[key] for key in sorted(selected)]:
            values = [run[3] for run in runs]
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.bar(labels, means, yerr=errors, capsize=4, color="#3978a8")
        ax.set(title=f"Final test accuracy — {title_suffix}", ylabel="Test accuracy", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(target / f"final_accuracy_{attack}.png", dpi=180)
        fig.savefig(target / f"final_accuracy_{attack}.pdf")
        plt.close(fig)
        if context[-1] != "legacy":
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
            for axis, metric in zip(axes.flat, ("tpr", "fpr", "fnr", "detection_f1", "defense_activation_rate", "unnecessary_defense_rate")):
                for position, key in enumerate(sorted(selected)):
                    values = [run[4][metric] for run in selected[key] if run[4].get(metric) is not None]
                    if values:
                        axis.bar(position, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0, capsize=3)
                    else:
                        axis.text(position, 0.05, "N/A", ha="center")
                axis.set(title=metric.replace("_", " "), ylim=(0, 1.05), xticks=range(len(labels)), xticklabels=labels)
                axis.tick_params(axis="x", rotation=30, labelsize=8)
                axis.grid(axis="y", alpha=0.2)
            fig.suptitle(f"Detection and defense — {title_suffix}\nMean ± sample SD; static aggregators do not classify clients", fontsize=12)
            fig.savefig(target / f"security_metrics_{attack}.png", dpi=180)
            fig.savefig(target / f"security_metrics_{attack}.pdf")
            plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
