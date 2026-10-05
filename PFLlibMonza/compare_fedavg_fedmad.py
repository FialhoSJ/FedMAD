#!/usr/bin/env python3
"""Create article-style plots comparing matched FedAvg and FedMAD MNIST runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_MAD_RUN = RESULTS_DIR / "20261002_212544_mnist_20c_50r_seed42"
OUT_DIR = RESULTS_DIR / "comparison_MNIST_FedAvg_FedMAD"

BLUE = "#1f77b4"
ORANGE = "#d95f02"
GRAY = "#7f7f7f"
GRID = "#cccccc"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linestyle="--", linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors="#333333", width=0.7, length=3)


def discover_avg_run() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.glob("*_fedavg_mnist_20c_50r_seed42")
        if (path / "metrics.h5").is_file() and (path / "FedAvg_server.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No completed FedAvg run found. Run ./run_fedavg_mnist_50r.sh first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    with h5py.File(run_dir / "metrics.h5", "r") as h5:
        data = {
            "accuracy": np.asarray(h5["rs_test_acc"], dtype=float),
            "loss": np.asarray(h5["rs_train_loss"], dtype=float),
            "seconds": np.asarray(h5["rs_train_time"], dtype=float),
        }
    lengths = {len(values) for values in data.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) != 50:
        raise ValueError(f"Expected 50 rounds of metrics in {run_dir}: {lengths}")
    return data


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
    total_correct = 0
    total_examples = 0
    total_loss = 0.0

    previous_cwd = Path.cwd()
    os.chdir(system_dir)
    try:
        with torch.no_grad():
            for client_id in range(20):
                dataset = read_client_data("MNIST", client_id, is_train=False)
                for features, labels in DataLoader(dataset, batch_size=256, shuffle=False):
                    outputs = model(features)
                    total_loss += loss_fn(outputs, labels).item()
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total_examples += labels.numel()
    finally:
        os.chdir(previous_cwd)

    return {
        "accuracy": total_correct / total_examples,
        "loss": total_loss / total_examples,
        "examples": total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mad-run", type=Path, default=DEFAULT_MAD_RUN)
    parser.add_argument("--avg-run", type=Path)
    args = parser.parse_args()

    mad_run = args.mad_run.resolve()
    avg_run = (args.avg_run or discover_avg_run()).resolve()
    mad = read_metrics(mad_run)
    avg = read_metrics(avg_run)
    mad_final = evaluate_checkpoint(mad_run, "MAD_server.pt")
    avg_final = evaluate_checkpoint(avg_run, "FedAvg_server.pt")
    rounds = np.arange(50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[plt.Figure] = []
    size = (6.48, 5.04)

    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    ax.plot(rounds, mad["accuracy"] * 100, color=ORANGE, linewidth=2.0, marker="o",
            markersize=3.0, markevery=5, label="FedMAD")
    ax.plot(rounds, avg["accuracy"] * 100, color=BLUE, linewidth=2.0, marker="s",
            markersize=3.0, markevery=5, label="FedAvg")
    for result, metrics, color, label in (
        (mad_final, mad, ORANGE, "FedMAD saved checkpoint"),
        (avg_final, avg, BLUE, "FedAvg saved checkpoint"),
    ):
        ax.plot([49, 50], [metrics["accuracy"][-1] * 100, result["accuracy"] * 100],
                color=color, linewidth=1.1, linestyle="--")
        ax.scatter([50], [result["accuracy"] * 100], color=color, marker="D", s=40,
                   edgecolors="white", linewidths=0.6, zorder=5, label=label)
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlim(0, 50)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    style_axis(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#bbbbbb")
    outputs.append(fig)

    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    ax.plot(rounds, mad["loss"], color=ORANGE, linewidth=2.0, marker="o",
            markersize=3.0, markevery=5, label="FedMAD")
    ax.plot(rounds, avg["loss"], color=BLUE, linewidth=2.0, marker="s",
            markersize=3.0, markevery=5, label="FedAvg")
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Training Loss")
    ax.set_xlim(0, 49)
    loss_floor = max(0.0, np.floor(min(mad["loss"].min(), avg["loss"].min()) * 5) / 5)
    loss_ceil = np.ceil(max(mad["loss"].max(), avg["loss"].max()) * 5) / 5
    ax.set_ylim(loss_floor, loss_ceil)
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#bbbbbb")
    outputs.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0), constrained_layout=True)
    categories = ["FedMAD", "FedAvg"]
    final_acc = [mad_final["accuracy"] * 100, avg_final["accuracy"] * 100]
    final_loss = [mad_final["loss"], avg_final["loss"]]
    total_minutes = [mad["seconds"].sum() / 60, avg["seconds"].sum() / 60]
    for ax, values, ylabel, title in (
        (axes[0], final_acc, "Test Accuracy (%)", "Final checkpoint accuracy"),
        (axes[1], final_loss, "Cross-Entropy Loss", "Final checkpoint loss"),
        (axes[2], total_minutes, "Minutes", "Training time"),
    ):
        bars = ax.bar(categories, values, color=[ORANGE, BLUE], width=0.62)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    outputs.append(fig)

    figure_names = ["fig_accuracy_comparison.png", "fig_loss_comparison.png", "fig_final_comparison.png"]
    for fig, filename in zip(outputs, figure_names):
        fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    with PdfPages(OUT_DIR / "comparison_figures.pdf") as pdf:
        for fig in outputs:
            pdf.savefig(fig, bbox_inches="tight")
    for fig in outputs:
        plt.close(fig)

    with (OUT_DIR / "round_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round_index", "fedmad_test_accuracy", "fedavg_test_accuracy",
                         "fedmad_train_loss", "fedavg_train_loss", "fedmad_seconds", "fedavg_seconds"])
        for index in rounds:
            writer.writerow([int(index), mad["accuracy"][index], avg["accuracy"][index],
                             mad["loss"][index], avg["loss"][index],
                             mad["seconds"][index], avg["seconds"][index]])

    summary = {
        "comparison": "FedMAD versus plain FedAvg",
        "matched_parameters": {
            "dataset": "MNIST, fixed 20-client Dirichlet non-IID partition (alpha=0.1)",
            "seed": 42,
            "clients": 20,
            "malicious_clients": [0, 1, 15, 17],
            "malicious_fraction": 0.2,
            "attack": "cyclic label flip y -> (y + 1) mod 10, active from round 0",
            "rounds": 50,
            "participation": "all clients per round",
            "model": "CNN",
            "local_epochs": 1,
            "batch_size": 10,
            "learning_rate": 0.005,
            "device": "CPU",
        },
        "FedMAD": {
            "run_directory": str(mad_run),
            "last_logged_accuracy_percent": float(mad["accuracy"][-1] * 100),
            "last_logged_round_index": 49,
            "final_checkpoint_accuracy_percent": float(mad_final["accuracy"] * 100),
            "final_checkpoint_loss": float(mad_final["loss"]),
            "final_test_examples": int(mad_final["examples"]),
            "last_train_loss": float(mad["loss"][-1]),
            "total_training_minutes": float(mad["seconds"].sum() / 60),
        },
        "FedAvg": {
            "run_directory": str(avg_run),
            "last_logged_accuracy_percent": float(avg["accuracy"][-1] * 100),
            "last_logged_round_index": 49,
            "final_checkpoint_accuracy_percent": float(avg_final["accuracy"] * 100),
            "final_checkpoint_loss": float(avg_final["loss"]),
            "final_test_examples": int(avg_final["examples"]),
            "last_train_loss": float(avg["loss"][-1]),
            "total_training_minutes": float(avg["seconds"].sum() / 60),
        },
        "delta_fedmad_minus_fedavg": {
            "final_accuracy_percentage_points": float((mad_final["accuracy"] - avg_final["accuracy"]) * 100),
            "final_checkpoint_loss": float(mad_final["loss"] - avg_final["loss"]),
        },
        "limitations": [
            "One seed and one fixed non-IID partition; results describe this run, not a statistical average.",
            "Both methods are evaluated under the same simulated label-flipping attack.",
            "Final checkpoint evaluation pools all 20 clients' test examples; logged per-round accuracy follows the framework metric.",
            "Runtime is affected by host load and is descriptive rather than a controlled benchmark.",
        ],
    }
    (OUT_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    mad_acc = mad_final["accuracy"] * 100
    avg_acc = avg_final["accuracy"] * 100
    report = f"""# Comparação: FedMAD e FedAvg

## Configuração pareada

- MNIST com a mesma partição não IID Dirichlet (α=0,1), 20 clientes e semente 42.
- Quatro clientes maliciosos (20%): 0, 1, 15 e 17.
- Ataque de inversão cíclica de rótulos `y → (y + 1) mod 10` ativo desde a rodada 0.
- CNN, 1 época local, batch 10, learning rate 0,005, CPU e 50 rodadas.
- FedAvg usa média ponderada simples, sem quarentena nem filtros extras.

## Resultados desta execução

| Medida | FedMAD | FedAvg | Diferença FedMAD − FedAvg |
|---|---:|---:|---:|
| Acurácia do checkpoint final (teste, todos os clientes) | {mad_acc:.2f}% | {avg_acc:.2f}% | {(mad_acc - avg_acc):+.2f} p.p. |
| Loss do checkpoint final | {mad_final['loss']:.4f} | {avg_final['loss']:.4f} | {mad_final['loss'] - avg_final['loss']:+.4f} |
| Acurácia registrada na avaliação da rodada 49 | {mad['accuracy'][-1] * 100:.2f}% | {avg['accuracy'][-1] * 100:.2f}% | {(mad['accuracy'][-1] - avg['accuracy'][-1]) * 100:+.2f} p.p. |
| Tempo total das rodadas | {mad['seconds'].sum() / 60:.1f} min | {avg['seconds'].sum() / 60:.1f} min | — |

As curvas usam os valores registrados pelo framework por rodada. O marcador de checkpoint final é calculado após a agregação da rodada 49, avaliando o mesmo modelo sobre os {mad_final['examples']:,} exemplos de teste.

## Arquivos

- `fig_accuracy_comparison.png`: acurácia por rodada e checkpoints finais.
- `fig_loss_comparison.png`: perda média de treino por rodada.
- `fig_final_comparison.png`: acurácia e loss finais e tempo observado.
- `comparison_figures.pdf`: as três figuras em um PDF.
- `round_metrics.csv`: séries numéricas por rodada.
- `comparison_summary.json`: configuração e resumo estruturado.

## Limites

É uma comparação concreta de uma semente e uma partição; não estima variabilidade entre execuções. Ambos os métodos enfrentam o mesmo ataque. O tempo depende da carga do computador e serve como registro desta execução, não como benchmark controlado.
"""
    (OUT_DIR / "analysis.md").write_text(report, encoding="utf-8")
    print(f"Comparison artifacts saved under: {OUT_DIR}")
    print(f"FedMAD final accuracy: {mad_acc:.2f}%")
    print(f"FedAvg final accuracy: {avg_acc:.2f}%")


if __name__ == "__main__":
    main()
