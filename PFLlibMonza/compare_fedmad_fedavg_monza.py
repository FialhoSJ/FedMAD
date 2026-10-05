#!/usr/bin/env python3
"""Generate matched FedMAD/FedAvg/MONZA MNIST comparison figures and report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-fedmad-monza")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_MAD_RUN = RESULTS_DIR / "20261002_212544_mnist_20c_50r_seed42"
OUT_DIR = RESULTS_DIR / "comparison_MNIST_FedMAD_FedAvg_MONZA"

MALICIOUS_IDS = [0, 1, 15, 17]
METHODS = [
    {"key": "fedmad", "label": "FedMAD", "color": "#d95f02",
     "default_run": DEFAULT_MAD_RUN, "pattern": None, "checkpoint": "MAD_server.pt"},
    {"key": "fedavg", "label": "FedAvg", "color": "#1f77b4",
     "default_run": None, "pattern": "*_fedavg_mnist_20c_50r_seed42", "checkpoint": "FedAvg_server.pt"},
    {"key": "monza", "label": "MONZA", "color": "#2ca02c",
     "default_run": None, "pattern": "*_monza_mnist_20c_50r_seed42", "checkpoint": "FedAvg_server.pt"},
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#cccccc", linestyle="--", linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors="#333333", width=0.7, length=3)


def find_run(spec: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
    elif spec["default_run"] is not None:
        path = spec["default_run"].resolve()
    else:
        candidates = [
            candidate for candidate in RESULTS_DIR.glob(spec["pattern"])
            if (candidate / "metrics.h5").is_file()
            and (candidate / spec["checkpoint"]).is_file()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No completed {spec['label']} run found. Run the corresponding runner first."
            )
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    if not (path / "metrics.h5").is_file() or not (path / spec["checkpoint"]).is_file():
        raise FileNotFoundError(f"Missing metrics/checkpoint for {spec['label']}: {path}")
    return path


def read_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    with h5py.File(run_dir / "metrics.h5", "r") as h5:
        values = {
            "accuracy": np.asarray(h5["rs_test_acc"], dtype=float),
            "loss": np.asarray(h5["rs_train_loss"], dtype=float),
            "seconds": np.asarray(h5["rs_train_time"], dtype=float),
        }
    lengths = {len(series) for series in values.values()}
    if lengths != {50}:
        raise ValueError(f"Expected 50 recorded rounds in {run_dir}; got {lengths}")
    return values


def evaluate_checkpoint(run_dir: Path, checkpoint_name: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    system_dir = PROJECT_DIR / "system"
    sys.path.insert(0, str(system_dir))
    from utils.data_utils import read_client_data

    torch.set_num_threads(1)
    model = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    examples = 0
    loss_total = 0.0
    previous_cwd = Path.cwd()
    os.chdir(system_dir)
    try:
        with torch.no_grad():
            for client_id in range(20):
                dataset = read_client_data("MNIST", client_id, is_train=False)
                for features, labels in DataLoader(dataset, batch_size=256, shuffle=False):
                    logits = model(features)
                    loss_total += loss_fn(logits, labels).item()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    examples += labels.numel()
    finally:
        os.chdir(previous_cwd)
    return {"accuracy": correct / examples, "loss": loss_total / examples, "examples": examples}


def parse_monza_filter_events(run_dir: Path) -> dict:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return {"available": False}
    content = log_path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"-------------Round number:\s*(\d+)-------------", content)
    events = []
    round_ids = []
    l2_fallback_rounds = []
    for index in range(1, len(parts), 2):
        round_number = int(parts[index])
        round_text = parts[index + 1]
        if "Gradiente L2 calculado para o cliente" in round_text:
            l2_fallback_rounds.append(round_number)
        found = [int(client_id) for client_id in
                 re.findall(r"Removing client (\d+) with score", round_text)]
        if found:
            round_ids.extend([round_number] * len(found))
            events.extend(found)
    flagged_unique = sorted(set(events))
    malicious_flag_events = sum(client_id in MALICIOUS_IDS for client_id in events)
    benign_flag_events = len(events) - malicious_flag_events
    return {
        "available": True,
        "method": "Count of per-round MONZA score-threshold removals parsed from the run log",
        "filter_events": len(events),
        "malicious_filter_events": malicious_flag_events,
        "benign_filter_events": benign_flag_events,
        "unique_flagged_clients": flagged_unique,
        "unique_malicious_clients_flagged": sorted(set(flagged_unique) & set(MALICIOUS_IDS)),
        "unique_benign_clients_flagged": sorted(set(flagged_unique) - set(MALICIOUS_IDS)),
        "rounds_with_filter_events": sorted(set(round_ids)),
        "l2_fallback_rounds": len(l2_fallback_rounds),
        "l2_fallback_round_ids": l2_fallback_rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mad-run", type=Path)
    parser.add_argument("--avg-run", type=Path)
    parser.add_argument("--monza-run", type=Path)
    args = parser.parse_args()
    explicit = {"fedmad": args.mad_run, "fedavg": args.avg_run, "monza": args.monza_run}

    results: dict[str, dict] = {}
    for spec in METHODS:
        run_dir = find_run(spec, explicit[spec["key"]])
        results[spec["key"]] = {
            "spec": spec,
            "run_dir": run_dir,
            "series": read_metrics(run_dir),
            "final": evaluate_checkpoint(run_dir, spec["checkpoint"]),
        }
    results["monza"]["filter_events"] = parse_monza_filter_events(results["monza"]["run_dir"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(50)
    figures: list[plt.Figure] = []

    fig, ax = plt.subplots(figsize=(6.48, 5.04), constrained_layout=True)
    for spec in METHODS:
        item = results[spec["key"]]
        ax.plot(rounds, item["series"]["accuracy"] * 100, color=spec["color"],
                linewidth=2.0, marker={"fedmad": "o", "fedavg": "s", "monza": "^"}[spec["key"]],
                markersize=3.0, markevery=5, label=spec["label"])
        final_acc = item["final"]["accuracy"] * 100
        ax.plot([49, 50], [item["series"]["accuracy"][-1] * 100, final_acc],
                color=spec["color"], linewidth=1.1, linestyle="--")
        ax.scatter([50], [final_acc], color=spec["color"], marker="D", s=38,
                   edgecolors="white", linewidths=0.6, zorder=5,
                   label=f"{spec['label']} final checkpoint")
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(0, 50)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    style_axis(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#bbbbbb")
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(6.48, 5.04), constrained_layout=True)
    for spec in METHODS:
        item = results[spec["key"]]
        ax.plot(rounds, item["series"]["loss"], color=spec["color"], linewidth=2.0,
                marker={"fedmad": "o", "fedavg": "s", "monza": "^"}[spec["key"]],
                markersize=3.0, markevery=5, label=spec["label"])
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Training Loss")
    ax.set_xlim(0, 49)
    loss_floor = max(0.0, np.floor(min(item["series"]["loss"].min()
                                       for item in results.values()) * 5) / 5)
    loss_ceiling = np.ceil(max(item["series"]["loss"].max()
                               for item in results.values()) * 5) / 5
    ax.set_ylim(loss_floor, loss_ceiling)
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#bbbbbb")
    figures.append(fig)

    labels = [spec["label"] for spec in METHODS]
    colors = [spec["color"] for spec in METHODS]
    final_acc = [results[spec["key"]]["final"]["accuracy"] * 100 for spec in METHODS]
    final_loss = [results[spec["key"]]["final"]["loss"] for spec in METHODS]
    elapsed_min = [results[spec["key"]]["series"]["seconds"].sum() / 60 for spec in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0), constrained_layout=True)
    for ax, values, ylabel, title in (
        (axes[0], final_acc, "Test Accuracy (%)", "Final checkpoint accuracy"),
        (axes[1], final_loss, "Cross-Entropy Loss", "Final checkpoint loss"),
        (axes[2], elapsed_min, "Minutes", "Recorded training time"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.65)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    figures.append(fig)

    image_names = ["fig_accuracy_comparison.png", "fig_loss_comparison.png", "fig_final_comparison.png"]
    for figure, filename in zip(figures, image_names):
        figure.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    with PdfPages(OUT_DIR / "comparison_figures.pdf") as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")
    for figure in figures:
        plt.close(figure)

    csv_path = OUT_DIR / "round_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round_index"] + [f"{spec['key']}_{metric}" for metric in
                                           ("accuracy", "loss", "seconds") for spec in METHODS])
        for round_index in rounds:
            row = [int(round_index)]
            for metric in ("accuracy", "loss", "seconds"):
                row.extend(float(results[spec["key"]]["series"][metric][round_index])
                           for spec in METHODS)
            writer.writerow(row)

    summary = {
        "comparison": "FedMAD versus pure FedAvg versus MONZA defense over FedAvg",
        "matched_parameters": {
            "dataset": "MNIST, same fixed 20-client Dirichlet non-IID partition (alpha=0.1)",
            "seed": 42,
            "clients": 20,
            "malicious_clients": MALICIOUS_IDS,
            "malicious_fraction": 0.2,
            "attack": "cyclic label flip y -> (y + 1) mod 10, active from round 0",
            "rounds": 50,
            "participation": "all available clients per round; MONZA may quarantine clients",
            "model": "CNN",
            "local_epochs": 1,
            "batch_size": 10,
            "learning_rate": 0.005,
            "device": "CPU",
        },
        "methods": {},
        "MONZA_filtering": results["monza"]["filter_events"],
        "interpretation_limit": "Single seed and partition; metrics describe these runs, not a statistical comparison across seeds.",
    }
    for spec in METHODS:
        item = results[spec["key"]]
        summary["methods"][spec["label"]] = {
            "run_directory": str(item["run_dir"]),
            "aggregation": {
                "FedMAD": "adaptive risk-based robust aggregation",
                "FedAvg": "plain sample-weighted average, -cc 5",
                "MONZA": "score-based client filtering/quarantine, -cc 3, then sample-weighted average",
            }[spec["label"]],
            "last_logged_test_accuracy_percent": float(item["series"]["accuracy"][-1] * 100),
            "final_checkpoint_accuracy_percent": float(item["final"]["accuracy"] * 100),
            "final_checkpoint_loss": float(item["final"]["loss"]),
            "final_test_examples": int(item["final"]["examples"]),
            "last_training_loss": float(item["series"]["loss"][-1]),
            "total_training_minutes": float(item["series"]["seconds"].sum() / 60),
        }

    (OUT_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metric_rows = []
    for spec in METHODS:
        item = results[spec["key"]]
        metric_rows.append(
            f"| {spec['label']} | {item['final']['accuracy'] * 100:.2f}% | "
            f"{item['final']['loss']:.4f} | {item['series']['seconds'].sum() / 60:.2f} min |"
        )
    event_summary = results["monza"]["filter_events"]
    monza_detection_note = (
        f"- MONZA score filter events: {event_summary.get('filter_events', 0)} total; "
        f"{event_summary.get('malicious_filter_events', 0)} from malicious IDs and "
        f"{event_summary.get('benign_filter_events', 0)} from benign IDs. "
        f"Unique malicious IDs flagged: {event_summary.get('unique_malicious_clients_flagged', [])}; "
        f"unique benign IDs flagged: {event_summary.get('unique_benign_clients_flagged', [])}.\n"
        f"- Score calculation used the L2 parameter-norm fallback in "
        f"{event_summary.get('l2_fallback_rounds', 0)} rounds; the code switches to this "
        f"path when the cosine-similarity matrix standard deviation is below 0.01."
        if event_summary.get("available") else "- MONZA filter-event log was unavailable."
    )
    report = f"""# Comparação FedMAD, FedAvg e MONZA

## Configuração pareada

- MNIST na mesma partição não IID Dirichlet (α=0,1), 20 clientes, semente 42.
- Clientes maliciosos 0, 1, 15 e 17 (20%), com label flipping cíclico `y → (y + 1) mod 10` desde a rodada 0.
- CNN, 1 época local, batch 10, learning rate 0,005, CPU, 50 rodadas e participação integral dos clientes disponíveis.
- FedAvg é a média ponderada sem defesa (`-cc 5`). MONZA usa o filtro por scores e quarentena do `serveravg.py` (`-cc 3`) antes de agregar com pesos por tamanho local.

## Acurácia do checkpoint final

Os checkpoints foram avaliados após a agregação final sobre os mesmos {results['fedmad']['final']['examples']:,} exemplos de teste.

| Método | Acurácia de teste | Loss de teste | Tempo registrado |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

Diferença FedMAD − FedAvg: **{(results['fedmad']['final']['accuracy'] - results['fedavg']['final']['accuracy']) * 100:+.2f} p.p.**.  
Diferença MONZA − FedAvg: **{(results['monza']['final']['accuracy'] - results['fedavg']['final']['accuracy']) * 100:+.2f} p.p.**.

## Detecção MONZA

{monza_detection_note}

Essas contagens vêm das decisões individuais do limiar no log. Elas são contagens de eventos de filtragem, não uma FPR/TPR estatística por cliente independente.

## Arquivos

- `fig_accuracy_comparison.png`: acurácia por rodada e acurácia final dos checkpoints.
- `fig_loss_comparison.png`: loss de treino por rodada.
- `fig_final_comparison.png`: acurácia final, loss de teste e tempo observado.
- `comparison_figures.pdf`: as três figuras.
- `round_metrics.csv`: séries de métricas para as 50 rodadas.
- `comparison_summary.json`: configuração e métricas estruturadas.

## Limites

É uma comparação de uma semente e uma partição. MONZA pontua as atualizações por similaridade de cosseno e exclui/quarentena as pontuações abaixo do limiar; sob heterogeneidade forte, clientes benignos também podem parecer discrepantes. Os resultados medem este ataque e esta execução, sem provar significância estatística ou eficácia universal. Tempos dependem da carga da máquina e são descritivos.
"""
    (OUT_DIR / "analysis.md").write_text(report, encoding="utf-8")

    print(f"Three-method comparison saved under: {OUT_DIR}")
    for spec in METHODS:
        final_percent = results[spec["key"]]["final"]["accuracy"] * 100
        print(f"{spec['label']} final checkpoint accuracy: {final_percent:.2f}%")
    print(f"MONZA score filter events: {event_summary.get('filter_events', 0)}")


if __name__ == "__main__":
    main()
