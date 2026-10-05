"""Plot paired FedMAD comparison runs listed in a run manifest."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_run(project: Path, row: dict):
    pattern = f"{row['dataset']}_{row['algorithm']}_*_{row['run_id']}_0.h5"
    candidates = list((project / "results").glob(pattern))
    if len(candidates) != 1:
        raise ValueError(f"{row['run_id']}: expected one HDF5 file, found {len(candidates)}")
    with h5py.File(candidates[0], "r") as h5:
        curve = np.asarray(h5["rs_test_acc"], dtype=float)
    final_path = Path(row["final_evaluation"])
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {}
    return curve, final.get("accuracy", curve[-1] if len(curve) else np.nan)


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
        key = (row["attack"], row["method"])
        curve, final_accuracy = read_run(project, row)
        groups.setdefault(key, []).append((int(row["seed"]), curve, float(final_accuracy)))
    if not groups:
        raise SystemExit("Manifest has no completed runs")

    attacks = sorted({key[0] for key in groups})
    for attack in attacks:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for (group_attack, method), runs in sorted(groups.items()):
            if group_attack != attack:
                continue
            length = min(len(curve) for _, curve, _ in runs)
            values = np.asarray([curve[:length] for _, curve, _ in runs])
            x = np.arange(length)
            avg = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(length)
            ax.plot(x, avg, marker="o", markersize=3, label=f"{method} (n={len(runs)})")
            if len(runs) > 1:
                ax.fill_between(x, avg - spread, avg + spread, alpha=0.16)
        ax.set(title=f"Test accuracy by round — {attack}", xlabel="Evaluation index", ylabel="Test accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(output / f"accuracy_by_round_{attack}.png", dpi=180)
        plt.close(fig)

        methods = sorted(method for a, method in groups if a == attack)
        means, errors = [], []
        for method in methods:
            values = [score for _, _, score in groups[(attack, method)]]
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.bar(methods, means, yerr=errors, capsize=4, color="#3978a8")
        ax.set(title=f"Final test accuracy — {attack}", ylabel="Test accuracy", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(output / f"final_accuracy_{attack}.png", dpi=180)
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
