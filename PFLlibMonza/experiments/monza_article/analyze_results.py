#!/usr/bin/env python3
"""Summarize paired runs and write paper-style curves, tables, and a PDF report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["fedmad", "fedavg", "monza", "fedavg-clean"]
LABELS = {
    "fedmad": "FedMAD",
    "fedavg": "FedAvg (sob ataque)",
    "monza": "MONZA",
    "fedavg-clean": "FedAvg limpo",
}
COLORS = {
    "fedmad": "#d95f02",
    "fedavg": "#1f77b4",
    "monza": "#2ca02c",
    "fedavg-clean": "#777777",
}
PAPER_TABLE4 = {
    "MONZA_MNIST_A02": {"monza": (97.11, 0.7, 7.5), "fedavg": (9.50, None, None), "clean": (97.25, None, None)},
    "MONZA_Cifar10_A02": {"monza": (47.58, 3.6, 7.68), "fedavg": (10.11, None, None), "clean": (50.07, None, None)},
    "MONZA_Cifar100_A02": {"monza": (22.64, 36.8, 7.4), "fedavg": (0.99, None, None), "clean": (25.35, None, None)},
}
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
                 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
                 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
                 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
                 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}


def mean_ci(values):
    data = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(data):
        return None, None, 0
    average = float(data.mean())
    if len(data) < 2:
        return average, None, len(data)
    critical = T_CRITICAL_95.get(len(data) - 1, 1.96)
    half_width = float(critical * data.std(ddof=1) / np.sqrt(len(data)))
    return average, half_width, len(data)


def fmt_ci(value, ci, scale=1.0, digits=2):
    if value is None:
        return "—"
    main = f"{value * scale:.{digits}f}"
    return main if ci is None else f"{main} ± {ci * scale:.{digits}f}"


def load_runs(root: Path):
    grouped = defaultdict(list)
    for path in sorted(root.glob("**/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path)
        grouped[(record["dataset"], record["method"])].append(record)
    if not grouped:
        raise FileNotFoundError(f"No metrics.json files under {root}")
    for records in grouped.values():
        records.sort(key=lambda record: record["seed"])
    return grouped


def curve(records, field, section="per_round"):
    arrays = [np.asarray(record[section][field], dtype=float) for record in records]
    length = min(len(array) for array in arrays)
    data = np.stack([array[:length] for array in arrays])
    if not np.isfinite(data).any():
        return np.full(length, np.nan), None
    means = np.nanmean(data, axis=0)
    if data.shape[0] < 2:
        return means, None
    critical = T_CRITICAL_95.get(data.shape[0] - 1, 1.96)
    half_width = critical * np.nanstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    return means, half_width


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#d0d0d0", linestyle="--", linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#333333", width=0.7, length=3)


def line_panel(ax, grouped, dataset, method_keys, field, label, x_end=None,
               percent=False, y_max=None):
    clipped = False
    for method in method_keys:
        records = grouped.get((dataset, method), [])
        if not records:
            continue
        means, ci = curve(records, field)
        if not np.isfinite(means).any():
            continue
        end = len(means) if x_end is None else min(len(means), x_end + 1)
        x = np.arange(end)
        values = means[:end] * (100 if percent else 1)
        if y_max is not None:
            clipped = clipped or bool(np.any(values > y_max))
            values = np.minimum(values, y_max)
        ax.plot(x, values, color=COLORS[method], linewidth=1.9,
                label=LABELS[method])
        if ci is not None:
            spread = ci[:end] * (100 if percent else 1)
            ax.fill_between(x, values - spread, values + spread,
                            color=COLORS[method], alpha=0.14, linewidth=0)
    ax.set_xlabel("Rodadas de comunicação")
    ax.set_ylabel(label)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if clipped:
        ax.text(
            0.99, 0.98,
            f"Valores acima de {y_max:g} limitados ao topo; consulte metrics.json para os valores completos",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="#444444",
        )
    style(ax)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#bbbbbb")


def seed_context(grouped, dataset, method_keys):
    seeds = sorted({
        record["seed"]
        for method in method_keys
        for record in grouped.get((dataset, method), [])
    })
    if len(seeds) == 1:
        return f"seed {seeds[0]} (n=1; IC95% não estimável)"
    return f"média ± IC 95% ({len(seeds)} seeds)"


def aggregate_rows(grouped, datasets):
    rows = []
    for dataset in datasets:
        for method in METHODS:
            records = grouped.get((dataset, method), [])
            if not records:
                continue
            values = {
                "best_accuracy": [record.get("max_test_accuracy") for record in records],
                "last_evaluated_accuracy": [record.get("final_recorded_accuracy") for record in records],
                "checkpoint_accuracy": [record.get("final_checkpoint_test", {}).get("accuracy") for record in records],
                "last_train_loss": [record.get("final_recorded_train_loss") for record in records],
                "fpr": [record.get("security_rounds", [{}])[-1].get("fpr") for record in records],
                "frr": [record.get("security_rounds", [{}])[-1].get("frr") for record in records],
                "total_round_seconds": [record.get("total_round_wall_seconds") for record in records],
                "client_seconds_per_call": [record.get("average_client_train_seconds_per_call") for record in records],
                "estimated_mflops_per_second": [record.get("estimated_training_mflops_per_second") for record in records],
            }
            row = {"dataset": dataset, "method": method, "label": LABELS[method], "n": len(records)}
            for metric, observations in values.items():
                average, half_width, count = mean_ci(observations)
                row[metric] = average
                row[f"{metric}_ci95"] = half_width
                if count != len(records) and metric in {"fpr", "frr"}:
                    row[f"{metric}_n"] = count
            row["seeds"] = ",".join(str(record["seed"]) for record in records)
            rows.append(row)
    return rows


def extract_fedmad_diagnostics(record):
    log_path = Path(record["_path"]).with_name("detection_log.json")
    if not log_path.is_file():
        return None
    rounds = json.loads(log_path.read_text(encoding="utf-8"))
    malicious = set(map(int, record.get("malicious_client_ids", [])))
    all_clients = set(range(int(record.get("num_clients", 0))))
    benign = all_clients - malicious
    rows = []
    for item in rounds:
        levels = {int(cid): level for cid, level in item.get("client_levels", {}).items()}
        risks = {int(cid): float(score) for cid, score in item.get("client_risks", {}).items()}
        blocked = {
            int(cid) for cid, duration in item.get("quarantine_after", {}).items()
            if int(duration) > 0
        }
        high = blocked | {cid for cid, level in levels.items() if level == "HIGH"}
        med_high = blocked | {cid for cid, level in levels.items() if level in {"MEDIUM", "HIGH"}}
        rows.append({
            "round": int(item["round"]),
            "mean_risk_malicious": float(np.mean([risks[cid] for cid in malicious if cid in risks])) if any(cid in risks for cid in malicious) else None,
            "mean_risk_benign": float(np.mean([risks[cid] for cid in benign if cid in risks])) if any(cid in risks for cid in benign) else None,
            "high_recall": len(high & malicious) / len(malicious) if malicious else None,
            "high_fpr": len(high & benign) / len(benign) if benign else None,
            "medium_high_recall": len(med_high & malicious) / len(malicious) if malicious else None,
            "medium_high_fpr": len(med_high & benign) / len(benign) if benign else None,
            "round_level": item.get("round_level"),
            "chosen_defense": item.get("chosen_defense"),
            "rollback": bool(item.get("rollback", False)),
            "rejected_candidates": sum(not outcome.get("accepted", False) for outcome in item.get("defense_attempts", [])),
            "attempted_candidates": len(item.get("defense_attempts", [])),
        })
    return rows


def write_fedmad_diagnostics(output: Path, grouped, datasets):
    report_rows = []
    run_rows = []
    for dataset in datasets:
        runs = grouped.get((dataset, "fedmad"), [])
        all_series = []
        for record in runs:
            series = extract_fedmad_diagnostics(record)
            if not series:
                continue
            all_series.append(series)
            final = series[-1]
            decisions = {}
            for item in series:
                name = item["chosen_defense"] or "rollback"
                decisions[name] = decisions.get(name, 0) + 1
            rejected = sum(item["rejected_candidates"] for item in series)
            attempted = sum(item["attempted_candidates"] for item in series)
            run_rows.append({
                "dataset": dataset,
                "seed": record["seed"],
                **final,
                "rollback_fraction": float(np.mean([item["rollback"] for item in series])),
                "candidate_rejection_fraction": rejected / attempted if attempted else 0.0,
                "chosen_defense_counts": json.dumps(decisions, sort_keys=True),
            })

        if not all_series:
            continue
        length = min(len(series) for series in all_series)
        risk_figure, risk_ax = plt.subplots(figsize=(6.48, 4.2), constrained_layout=True)
        for field, label, color in (
            ("mean_risk_malicious", "Clientes maliciosos", "#d95f02"),
            ("mean_risk_benign", "Clientes benignos", "#1f77b4"),
        ):
            data = np.asarray([[series[index][field] for index in range(length)] for series in all_series], dtype=float)
            means = np.nanmean(data, axis=0)
            x = np.arange(length)
            risk_ax.plot(x, means, color=color, linewidth=1.9, label=label)
            if len(all_series) > 1:
                critical = T_CRITICAL_95.get(len(all_series) - 1, 1.96)
                spread = critical * np.nanstd(data, axis=0, ddof=1) / np.sqrt(len(all_series))
                risk_ax.fill_between(x, means - spread, means + spread, color=color, alpha=0.14)
        risk_ax.axhline(0.35, color="#777777", linestyle="--", linewidth=0.9, label="Limiar LOW/MEDIUM (0,35)")
        risk_ax.axhline(0.65, color="#555555", linestyle=":", linewidth=1.0, label="Limiar MEDIUM/HIGH (0,65)")
        risk_ax.set_xlabel("Rodadas de comunicação")
        risk_ax.set_ylabel("Risco médio previsto")
        risk_ax.set_title(f"{dataset}: risco previsto pelo FedMAD", fontsize=12)
        style(risk_ax)
        risk_ax.legend(frameon=True, framealpha=0.95, edgecolor="#bbbbbb")
        base = output / f"{dataset}_fedmad_risk"
        risk_figure.savefig(base.with_suffix(".png"), dpi=220)
        risk_figure.savefig(base.with_suffix(".pdf"))
        plt.close(risk_figure)

        final_rows = [series[-1] for series in all_series]
        summary = {"dataset": dataset, "n": len(final_rows)}
        for field in ("mean_risk_malicious", "mean_risk_benign", "high_recall",
                      "high_fpr", "medium_high_recall", "medium_high_fpr"):
            mean, ci, _ = mean_ci([row[field] for row in final_rows])
            summary[field] = mean
            summary[f"{field}_ci95"] = ci
        report_rows.append(summary)

    if run_rows:
        with (output / "fedmad_diagnostics.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)
    return report_rows


def write_markdown_report(report_path: Path, rows, environment, grouped, mad_rows):
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    seeds = sorted({
        record["seed"]
        for records in grouped.values()
        for record in records
    })
    rounds_argument = environment.get("rounds_argument")
    loop_updates = environment.get("loop_updates_in_pfl_lib")
    if rounds_argument is None:
        rounds_argument = max(
            (record.get("rounds_argument", 0)
             for records in grouped.values() for record in records),
            default=None,
        )
    if loop_updates is None and rounds_argument is not None:
        loop_updates = rounds_argument
    configured_clients = environment.get("num_clients") or max(
        (record.get("num_clients", 0)
         for records in grouped.values() for record in records),
        default=None,
    )
    malicious_clients = environment.get("malicious_clients")
    if malicious_clients is None:
        malicious_clients = max(
            (record.get("malicious_clients", 0)
             for records in grouped.values() for record in records),
            default=None,
        )
    scope = []
    if configured_clients is not None:
        scope.append(f"{configured_clients} clientes")
    if malicious_clients is not None and configured_clients:
        scope.append(
            f"{malicious_clients} maliciosos ({100 * malicious_clients / configured_clients:.1f}%)"
        )
    if rounds_argument is not None:
        pfl_rounds = environment.get("pfl_global_rounds_argument")
        pfl_detail = f", PFLlib -gr {pfl_rounds}" if pfl_rounds is not None else ""
        scope.append(f"{rounds_argument} rodadas ({loop_updates} atualizações{pfl_detail})")
    scope.append(f"seeds {seeds}")
    scope.append(f"datasets {environment.get('datasets', datasets)}")
    scope.append(f"métodos {environment.get('methods', methods)}")
    if environment.get("device"):
        scope.append(f"dispositivo {environment['device']}")

    hardware = []
    if environment.get("platform"):
        hardware.append(str(environment["platform"]))
    if environment.get("processor"):
        hardware.append(f"CPU {environment['processor']}")
    if environment.get("logical_cpu_count") is not None:
        hardware.append(f"{environment['logical_cpu_count']} processadores lógicos")
    if environment.get("torch"):
        hardware.append(f"PyTorch {environment['torch']}")
    if environment.get("cuda_available") is not None:
        hardware.append(f"CUDA disponível: {environment['cuda_available']}")

    if environment.get("attack_profile") == "article":
        attack_parameters = environment.get("attack_parameters", {})
        attack_description = (
            "- Perfil `article`: zero substitui os parâmetros por zeros; Random adiciona ruído Gaussiano "
            f"com SNR linear {attack_parameters.get('random_noise_linear_snr', 1.0)}; Shuffled-Layer "
            "permuta canais de saída separadamente em cada camada Conv/Linear; label flipping troca "
            f"a classe {attack_parameters.get('label_flip_source', 0)} pela classe "
            f"{attack_parameters.get('label_flip_target', 1)}. A tabela do artigo não informa o SNR nem "
            "o par de classes; esses valores são hipóteses explícitas do experimento. Um tipo de ataque "
            "é escolhido por cliente malicioso em cada rodada."
        )
    else:
        attack_description = (
            "- `-atk all` escolhe um tipo por cliente malicioso em cada rodada. O perfil `repository` "
            "usa os ataques implementados no projeto: parâmetros zero, substituição Uniform[0,1], "
            "permutação independente dos elementos de cada tensor e rotação cíclica dos rótulos."
        )

    lines = [
        "# Relatório da comparação",
        "",
        "Este relatório resume os resultados realmente encontrados nesta pasta. "
        "As referências publicadas do artigo aparecem separadas e não devem ser confundidas "
        "com esta execução piloto.",
        "",
        "Artigo: [MONZA: A Score System for Malicious Clients Detection](https://journals-sol.sbc.org.br/index.php/jisa/article/download/7098/4181/42713), DOI 10.5753/jisa.2026.7098.",
        "",
        ("Hardware registrado: " + "; ".join(hardware) + ".")
        if hardware else "Hardware desta execução não foi registrado.",
        "",
        "**Escopo executado:** " + "; ".join(scope) + ". "
        + ("É uma comparação com horizonte reduzido em relação às 500 rodadas do artigo."
           if rounds_argument is None or rounds_argument < 500
           else "A grade contém o número de rodadas, seeds e datasets do protocolo selecionado."),
        "",
        "## Comparação por conjunto de dados",
        "",
        "Acurácia melhor é maior; perda, FPR e FRR menores são melhores. FPR e FRR são frações convertidas para porcentagem. "
        "Para o FedAvg sem defesa, todos os participantes são aceitos: FPR=0% e FRR=100% sob ataque. "
        "No FedAvg limpo, FPR=0% e FRR fica indefinido porque não há cliente malicioso.",
        "",
        "| Dataset | Método | n | Melhor acurácia (%) | Acurácia no último ponto medido (%) | Loss final | FPR (%) | FRR (%) | Tempo total (min) | MFLOP/s estimado |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fpr = fmt_ci(row.get("fpr"), row.get("fpr_ci95"), scale=100, digits=2)
        frr = fmt_ci(row.get("frr"), row.get("frr_ci95"), scale=100, digits=2)
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['n']} | "
            f"{fmt_ci(row.get('best_accuracy'), row.get('best_accuracy_ci95'), scale=100)} | "
            f"{fmt_ci(row.get('last_evaluated_accuracy'), row.get('last_evaluated_accuracy_ci95'), scale=100)} | "
            f"{fmt_ci(row.get('last_train_loss'), row.get('last_train_loss_ci95'), digits=3)} | "
            f"{fpr} | {frr} | "
            f"{fmt_ci(row.get('total_round_seconds'), row.get('total_round_seconds_ci95'), scale=1/60, digits=1)} | "
            f"{fmt_ci(row.get('estimated_mflops_per_second'), row.get('estimated_mflops_per_second_ci95'), digits=1)} |"
        )
    lines += [
        "",
        "## Valores de referência publicados na Tabela 4",
        "",
        "| Dataset | MONZA: artigo (acc/FPR/FRR) | FedAvg atacado: artigo | FedAvg limpo: artigo |",
        "|---|---:|---:|---:|",
    ]
    for dataset, values in PAPER_TABLE4.items():
        monza = values["monza"]
        fedavg = values["fedavg"][0]
        clean = values["clean"][0]
        lines.append(
            f"| {dataset} | {monza[0]:.2f}% / {monza[1]:.2f}% / {monza[2]:.2f}% | "
            f"{fedavg:.2f}% | {clean:.2f}% |"
        )
    lines += [
        "",
        "## Como ler a comparação",
        "",
        "- Compare as curvas e os intervalos de confiança das mesmas sementes. Acurácia maior junto com FPR e FRR menores caracteriza uma defesa melhor neste cenário.",
        "- Uma diferença que muda de sinal entre datasets indica dependência da tarefa; média sobre os três datasets não deve esconder essa variação.",
        f"- O projeto executa MONZA como `FedAvg` com `-cc 3`. {environment.get('monza_implementation', 'FedMAD e MONZA usam o mesmo modelo CNN, partições, seeds e clientes maliciosos nesta comparação.')}",
        attack_description,
        "- A avaliação em lotes do modo padrão foi confrontada com a rotina original usando a mesma seed: acurácias e pesos finais coincidiram; a diferença máxima observada na cross-entropy foi 5,8×10⁻⁵ por acumulação float32. A AUC adicional é omitida no modo rápido.",
        "- O campo de FLOP/s é uma estimativa explícita deste relatório: conta MACs das convoluções/camadas lineares, considera três passagens equivalentes no treino e divide pelo tempo de treino medido. O artigo descreve uma fórmula simplificada cuja unidade publicada não fica dimensionalmente consistente; portanto compare esse número entre os métodos nesta máquina, não diretamente com MFLOP/s do artigo.",
        "- A Tabela 4 e o texto do artigo não coincidem em todos os resultados de CIFAR-10: a tabela informa 47,58% para MONZA e 50,07% para FedAvg limpo, enquanto a narrativa/final de curva descreve valores diferentes. A referência tabular é mantida acima e o conflito não é resolvido por suposição.",
        "- A acurácia das curvas vem da avaliação gravada pelo PFLlib; o `final_model.pt` também é avaliado ao final e a medida fica em `final_checkpoint_test` no JSON de cada execução.",
        "",
        "## Sinais internos do FedMAD",
        "",
        "O CSV e o gráfico `*_fedmad_risk` mostram risco médio previsto para clientes maliciosos e benignos, quantos chegam aos níveis MEDIUM/HIGH, defesas aceitas e rejeições do validador. Se o risco dos maliciosos ficar abaixo do limiar HIGH e o FRR permanecer alto, o primeiro ajuste a investigar é o limiar/paciência da quarentena; se o risco benigno subir junto, a correção deve separar deriva não IID de comportamento malicioso antes de reduzir o limiar.",
        "",
    ]
    if mad_rows:
        lines += [
            "| Dataset | n | Risco malicioso final | Risco benigno final | Recall HIGH | FPR HIGH | Recall MEDIUM+ | FPR MEDIUM+ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in mad_rows:
            lines.append(
                f"| {row['dataset']} | {row['n']} | "
                f"{fmt_ci(row['mean_risk_malicious'], row['mean_risk_malicious_ci95'], digits=3)} | "
                f"{fmt_ci(row['mean_risk_benign'], row['mean_risk_benign_ci95'], digits=3)} | "
                f"{fmt_ci(row['high_recall'], row['high_recall_ci95'], scale=100)} | "
                f"{fmt_ci(row['high_fpr'], row['high_fpr_ci95'], scale=100)} | "
                f"{fmt_ci(row['medium_high_recall'], row['medium_high_recall_ci95'], scale=100)} | "
                f"{fmt_ci(row['medium_high_fpr'], row['medium_high_fpr_ci95'], scale=100)} |"
            )
        lines.append("")
    lines.append("Arquivos PNG/PDF contêm as curvas de acurácia, loss, FPR/FRR, risco do FedMAD e custo computacional. Os dados por seed estão em `runs/<dataset>/<método>/seed_<n>/metrics.json` e `metrics.h5`.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Experiment directory containing environment.json and run folders.")
    args = parser.parse_args()
    root = args.results.resolve()
    grouped = load_runs(root)
    datasets = sorted({dataset for dataset, _ in grouped})
    env_path = root / "environment.json"
    environment = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = aggregate_rows(grouped, datasets)
    mad_rows = write_fedmad_diagnostics(output, grouped, datasets)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as target:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    for dataset in datasets:
        methods = [method for method in METHODS if grouped.get((dataset, method))]
        uncertainty_label = seed_context(grouped, dataset, methods)
        method_labels = ", ".join(LABELS[method] for method in methods)
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)
        line_panel(axes[0, 0], grouped, dataset, methods, "accuracy",
                   "Acurácia de teste (%)", x_end=300, percent=True)
        line_panel(axes[0, 1], grouped, dataset, methods, "training_cross_entropy",
                   "Cross-entropy de treino", x_end=500, y_max=500)
        line_panel(axes[1, 0], grouped, dataset, methods, "fpr",
                   "FPR (%)", x_end=500, percent=True)
        line_panel(axes[1, 1], grouped, dataset, methods, "frr",
                   "FRR (%)", x_end=500, percent=True)
        fig.suptitle(f"{dataset}: {method_labels} ({uncertainty_label})", fontsize=13)
        base = output / f"{dataset}_curves"
        fig.savefig(base.with_suffix(".png"), dpi=220)
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
        categories = methods
        for ax, key, source_key, title, scale, units in (
            (axes[0], "total_round_seconds", "total_round_wall_seconds", "Tempo total", 1 / 60, "min"),
            (axes[1], "client_seconds_per_call", "average_client_train_seconds_per_call", "Tempo médio por treino local", 1, "s/chamada"),
            (axes[2], "estimated_mflops_per_second", "estimated_training_mflops_per_second", "Custo estimado", 1, "MFLOP/s"),
        ):
            means, errors = [], []
            for method in categories:
                observations = [
                    record.get(source_key) for record in grouped.get((dataset, method), [])
                ]
                avg, ci, _ = mean_ci(observations)
                means.append(np.nan if avg is None else avg * scale)
                errors.append(0 if ci is None else ci * scale)
            x = np.arange(len(categories))
            bars = ax.bar(x, means, yerr=errors, capsize=3,
                          color=[COLORS[method] for method in categories],
                          edgecolor="white", linewidth=0.6)
            ax.set_xticks(x, [LABELS[method] for method in categories], rotation=18, ha="right")
            ax.set_ylabel(units)
            ax.set_title(title)
            for bar in bars:
                if np.isfinite(bar.get_height()):
                    ax.annotate(f"{bar.get_height():.2f}",
                                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)
            style(ax)
        fig.suptitle(f"{dataset}: custo experimental ({uncertainty_label})", fontsize=13)
        base = output / f"{dataset}_cost"
        fig.savefig(base.with_suffix(".png"), dpi=220)
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)

    write_markdown_report(output / "report.md", rows, environment, grouped, mad_rows)
    print(f"Analysis written to {output}")


if __name__ == "__main__":
    main()
