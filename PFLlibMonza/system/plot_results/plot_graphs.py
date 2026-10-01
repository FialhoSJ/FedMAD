import h5py
import json
import csv
import os
import re
from collections import defaultdict
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
})

RESULTS_DIR = Path(__file__).parents[2] / "results"
SYSTEM_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "output"
CSV_DIR = SYSTEM_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── helpers ───────────────────────────

def parse_h5_filename(fname: str):
    stem = Path(fname).stem
    parts = stem.split("_")
    data = parts[0]
    algo = parts[1]
    idx_cc = 2
    cc = int(parts[idx_cc])
    rate = int(parts[idx_cc + 1])
    nmc = int(parts[idx_cc + 2])
    trial = int(parts[-1])
    return dict(dataset=data, algorithm=algo, cc=cc,
                rate_client_fake=rate, n_client_malicious=nmc,
                trial=trial)


def load_h5(path):
    with h5py.File(path, 'r') as f:
        return {k: f[k][:] for k in f.keys()}


def load_all_h5():
    files = sorted(RESULTS_DIR.glob("*.h5"))
    data = []
    for fp in files:
        meta = parse_h5_filename(fp.name)
        meta['path'] = fp
        meta['label'] = f"{meta['dataset']} {meta['algorithm']} cc={meta['cc']} nmc={meta['n_client_malicious']} trial={meta['trial']}"
        meta['short'] = f"cc={meta['cc']} nmc={meta['n_client_malicious']}"
        data.append(meta)
    return data


def load_fpr_frr_csvs():
    files = sorted(CSV_DIR.glob("fpr_frr_results_*.csv"))
    data = []
    for fp in files:
        m = re.search(r'fpr_frr_results_(\d+)\.csv', fp.name)
        cc = int(m.group(1)) if m else 0
        rows = []
        with open(fp, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    'round': int(row['Round']),
                    'FPR': float(row['FPR']),
                    'FRR': float(row['FRR']),
                })
        data.append({'cc': cc, 'path': fp, 'rows': rows})
    return data


def _safe_fig(name):
    path = OUTPUT_DIR / name
    plt.savefig(path, bbox_inches='tight')
    print(f"  saved -> {path.name}")
    plt.close()


# ─────────────────────── 1. individual curves ──────────────────

def plot_individual_curve(dataset, field, ylabel, title_prefix):
    fig, ax = plt.subplots()
    for meta in dataset:
        vals = meta[field]
        if len(vals) == 0:
            continue
        ax.plot(vals, label=meta['label'], lw=1.2)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — all experiments")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _safe_fig(f"all_{field}.png")


def plot_all_individuals(dataset):
    for meta in dataset:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(meta['label'], fontsize=13, fontweight='bold')

        acc = meta['rs_test_acc']
        auc = meta['rs_test_auc']
        loss = meta['rs_train_loss']
        t = meta['rs_train_time']

        axes[0, 0].plot(acc)
        axes[0, 0].set_title("Test Accuracy")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].grid(True, alpha=0.3)

        if len(auc) > 0:
            axes[0, 1].plot(auc)
            axes[0, 1].set_title("Test AUC")
        else:
            axes[0, 1].text(0.5, 0.5, "AUC not available",
                            ha='center', va='center', transform=axes[0, 1].transAxes,
                            style='italic', color='gray')
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(loss, color='orange')
        axes[1, 0].set_title("Train Loss")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(t, color='green')
        axes[1, 1].set_title("Time per Round (s)")
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = meta['path'].stem + "_individual.png"
        _safe_fig(safe_name)


# ────────────────── 2. overlay comparison ──────────────────────

def plot_overlay_comparison(dataset):
    by_dataset = {}
    for meta in dataset:
        key = meta['dataset']
        by_dataset.setdefault(key, []).append(meta)

    for ds, items in by_dataset.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(f"Overlay — {ds}", fontsize=13, fontweight='bold')

        fields = [
            ('rs_test_acc', 'Test Accuracy', axes[0, 0], None),
            ('rs_test_auc', 'Test AUC', axes[0, 1], None),
            ('rs_train_loss', 'Train Loss', axes[1, 0], 'orange'),
            ('rs_train_time', 'Time / Round (s)', axes[1, 1], 'green'),
        ]

        for field, ylabel, ax, color in fields:
            for meta in items:
                vals = meta[field]
                if len(vals) == 0:
                    continue
                kw = dict(lw=1.2)
                if color:
                    kw['color'] = color
                ax.plot(vals, label=meta['short'], **kw)
            ax.set_title(ylabel)
            ax.set_xlabel("Round")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _safe_fig(f"overlay_{ds}.png")


# ──────────────────── 3. bar chart (best accuracy) ─────────────

def plot_best_acc_bar(dataset):
    items = []
    for meta in dataset:
        acc = meta['rs_test_acc']
        best = float(np.max(acc)) if len(acc) > 0 else 0
        last = float(acc[-1]) if len(acc) > 0 else 0
        items.append({**meta, 'best': best, 'last': last})

    if not items:
        return

    labels = [it['short'] for it in items]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    bars1 = ax.bar(x - width / 2, [it['best'] for it in items], width, label='Best', color='steelblue')
    bars2 = ax.bar(x + width / 2, [it['last'] for it in items], width, label='Final', color='lightcoral')

    ax.set_ylabel("Accuracy")
    ax.set_title("Best / Final Test Accuracy per experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    _safe_fig("best_accuracy_comparison.png")

    # also a summary table
    print("\nAccuracy summary:")
    print(f"{'Label':<25} {'Best':>8} {'Final':>8}")
    print("-" * 45)
    for it in items:
        print(f"{it['short']:<25} {it['best']:>8.4f} {it['last']:>8.4f}")


# ──────────────────── 4. total time bar ────────────────────────

def plot_total_time_bar(dataset):
    items = []
    for meta in dataset:
        t = meta['rs_train_time']
        total = float(np.sum(t)) if len(t) > 0 else 0
        items.append({**meta, 'total_time': total})

    if not items:
        return

    labels = [it['short'] for it in items]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    ax.bar(x, [it['total_time'] for it in items], color='mediumseagreen')
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Total Execution Time per experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for i, it in enumerate(items):
        ax.text(i, it['total_time'], f'{it["total_time"]:.0f}s',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    _safe_fig("total_time_comparison.png")


# ──────────────────── 5. FPR / FRR curves ──────────────────────

def plot_fpr_frr(fprfrr_data):
    if not fprfrr_data:
        print("  (no FPR/FRR CSV files found)")
        return

    for item in fprfrr_data:
        rows = item['rows']
        rounds = [r['round'] for r in rows]
        fpr = [r['FPR'] for r in rows]
        frr = [r['FRR'] for r in rows]

        fig, ax = plt.subplots()
        ax.plot(rounds, fpr, label='FPR', marker='.', lw=1.2)
        ax.plot(rounds, frr, label='FRR', marker='.', lw=1.2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Rate")
        ax.set_title(f"FPR / FRR — cc={item['cc']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _safe_fig(f"fpr_frr_cc{item['cc']}.png")

    # overlay all on same plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for item in fprfrr_data:
        rows = item['rows']
        rounds = [r['round'] for r in rows]
        fpr = [r['FPR'] for r in rows]
        frr = [r['FRR'] for r in rows]
        axes[0].plot(rounds, fpr, label=f"cc={item['cc']}", lw=1.2)
        axes[1].plot(rounds, frr, label=f"cc={item['cc']}", lw=1.2)
    axes[0].set_title("FPR overlay")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("FPR")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("FRR overlay")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("FRR")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _safe_fig("fpr_frr_overlay.png")

    # final FPR/FRR bar
    fig, ax = plt.subplots()
    labels = [f"cc={item['cc']}" for item in fprfrr_data]
    x = np.arange(len(labels))
    w = 0.35
    final_fpr = [item['rows'][-1]['FPR'] for item in fprfrr_data]
    final_frr = [item['rows'][-1]['FRR'] for item in fprfrr_data]
    ax.bar(x - w / 2, final_fpr, w, label='Final FPR', color='steelblue')
    ax.bar(x + w / 2, final_frr, w, label='Final FRR', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("Final FPR / FRR comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _safe_fig("fpr_frr_final_bar.png")


# ──────────────────── 6. summary dashboard ─────────────────────

def plot_summary_dashboard(dataset, fprfrr_data):
    """Single overview figure with best acc, total time, and avg time per round."""
    items = []
    for meta in dataset:
        acc = meta['rs_test_acc']
        t = meta['rs_train_time']
        items.append({
            'short': meta['short'],
            'best': float(np.max(acc)) if len(acc) > 0 else 0,
            'last': float(acc[-1]) if len(acc) > 0 else 0,
            'total_time': float(np.sum(t)) if len(t) > 0 else 0,
            'avg_time': float(np.mean(t)) if len(t) > 0 else 0,
        })

    if not items:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [it['short'] for it in items]
    x = np.arange(len(labels))

    ax = axes[0]
    ax.bar(x, [it['best'] for it in items], color='steelblue')
    ax.set_title("Best Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, [it['total_time'] for it in items], color='mediumseagreen')
    ax.set_title("Total Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(x, [it['avg_time'] for it in items], color='darkorange')
    ax.set_title("Avg Time / Round (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    if fprfrr_data:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        labels2 = [f"cc={item['cc']}" for item in fprfrr_data]
        x2 = np.arange(len(labels2))
        w2 = 0.35
        final_fpr = [item['rows'][-1]['FPR'] for item in fprfrr_data]
        final_frr = [item['rows'][-1]['FRR'] for item in fprfrr_data]
        ax2.bar(x2 - w2 / 2, final_fpr, w2, label='Final FPR', color='steelblue')
        ax2.bar(x2 + w2 / 2, final_frr, w2, label='Final FRR', color='lightcoral')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels2)
        ax2.set_ylabel("Rate")
        ax2.set_title("Final FPR / FRR")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        _safe_fig("fpr_frr_final_bar.png")

    plt.tight_layout()
    _safe_fig("summary_dashboard.png")


# ──────────────────── 7. nº rounds executed ───────────────────

def plot_num_rounds(dataset):
    items = [(meta['short'], len(meta['rs_test_acc']))
             for meta in dataset if len(meta['rs_test_acc']) > 0]
    if len(items) < 2:
        return
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    ax.bar(range(len(labels)), vals, color='cornflowerblue')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel("Rounds")
    ax.set_title("Number of rounds executed")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha='center', va='bottom')
    _safe_fig("num_rounds.png")


# ============================================================
#   FedMAD detection log graphs (from detection_log_*.json)
# ============================================================

def load_detection_logs():
    files = sorted(RESULTS_DIR.glob("detection_log_*.json"))
    logs = []
    for fp in files:
        with open(fp, 'r') as f:
            data = json.load(f)
        logs.append({"path": fp, "data": data, "name": fp.stem.replace("detection_log_", "")})
    return logs


def plot_agent_scores_heatmap(log):
    """Heatmap: rounds x clients, one per agent."""
    data = log["data"]
    agent_names = data[0]["agent_names"]
    client_ids_sorted = sorted({cid for round_data in data for cid in round_data["per_agent_scores"][agent_names[0]]})
    client_ids_sorted = [c for c in client_ids_sorted if c.isdigit()]
    client_ids_sorted.sort(key=int)

    rounds = [r["round"] for r in data]

    for aidx, aname in enumerate(agent_names):
        matrix = np.full((len(rounds), len(client_ids_sorted)), np.nan)
        for ri, rd in enumerate(data):
            scores = rd["per_agent_scores"].get(aname, {})
            for ci, cid in enumerate(client_ids_sorted):
                if cid in scores:
                    matrix[ri, ci] = scores[cid]

        fig, ax = plt.subplots(figsize=(max(8, len(client_ids_sorted) * 0.5), max(5, len(rounds) * 0.3)))
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(rounds)))
        ax.set_yticklabels(rounds, fontsize=7)
        ax.set_xticks(np.arange(len(client_ids_sorted)))
        ax.set_xticklabels(client_ids_sorted, fontsize=7, rotation=90)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Round")
        ax.set_title(f"Scores — {aname} ({log['name']})")
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("Anomaly Score")
        _safe_fig(f"heatmap_{aname}_{log['name']}.png")


def plot_flagged_per_round(log):
    """How many clients each agent flagged per round (score > 0.6)."""
    data = log["data"]
    agent_names = data[0]["agent_names"]

    rounds = [r["round"] for r in data]
    fig, ax = plt.subplots()
    for aname in agent_names:
        flagged = []
        for rd in data:
            scores = rd["per_agent_scores"].get(aname, {})
            n = sum(1 for v in scores.values() if v > 0.6)
            flagged.append(n)
        ax.plot(rounds, flagged, label=aname, marker='.', lw=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Clients flagged (score > 0.6)")
    ax.set_title(f"Clients flagged per agent — {log['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _safe_fig(f"flagged_per_round_{log['name']}.png")

    # Also show final-score flagged (actual removals)
    removed_counts = []
    for rd in data:
        removed_counts.append(len(rd.get("removed_clients", [])))
    if any(removed_counts):
        ax.plot(rounds, removed_counts, label='Actually removed', marker='x', lw=2, color='black', linestyle='--')
        ax.legend()
        _safe_fig(f"flagged_per_round_{log['name']}.png")  # overwrite with updated version


def plot_score_distribution(log):
    """Box-plot: scores by agent, split by malicious vs benign."""
    data = log["data"]
    agent_names = data[0]["agent_names"]

    malicious_scores = {an: [] for an in agent_names}
    benign_scores = {an: [] for an in agent_names}

    for rd in data:
        gt = {str(cid): int(m) for cid, m in zip(rd["client_ids"], rd["malicious_ground_truth"])}
        for aname in agent_names:
            scores = rd["per_agent_scores"].get(aname, {})
            for cid, sc in scores.items():
                if gt.get(cid, 0):
                    malicious_scores[aname].append(sc)
                else:
                    benign_scores[aname].append(sc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx, (title, score_dict) in enumerate([("Benign clients", benign_scores), ("Malicious clients", malicious_scores)]):
        ax = axes[idx]
        data_to_plot = [score_dict[an] if score_dict[an] else [0] for an in agent_names]
        bp = ax.boxplot(data_to_plot, labels=agent_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel("Anomaly Score")
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f"Score distribution by agent — {log['name']}", fontweight='bold')
    plt.tight_layout()
    _safe_fig(f"score_distribution_{log['name']}.png")


def plot_agent_agreement_heatmap(log):
    """Correlation matrix of agent scores."""
    data = log["data"]
    agent_names = data[0]["agent_names"]
    n_agents = len(agent_names)

    all_vectors = {an: [] for an in agent_names}
    for rd in data:
        for aname in agent_names:
            scores = rd["per_agent_scores"].get(aname, {})
            all_vectors[aname].extend(list(scores.values()))

    # truncate to same length
    min_len = min(len(v) for v in all_vectors.values())
    matrix = np.column_stack([all_vectors[an][:min_len] for an in agent_names])
    corr = np.corrcoef(matrix.T)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(np.arange(n_agents))
    ax.set_yticks(np.arange(n_agents))
    ax.set_xticklabels(agent_names, rotation=30, ha='right')
    ax.set_yticklabels(agent_names)
    for i in range(n_agents):
        for j in range(n_agents):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', fontsize=8)
    ax.set_title(f"Agent score correlation — {log['name']}")
    fig.colorbar(im, ax=ax, shrink=0.7)
    _safe_fig(f"agent_agreement_{log['name']}.png")


def plot_quarantine_evolution(log):
    """How many clients were in quarantine over rounds."""
    data = log["data"]
    rounds = [r["round"] for r in data]
    quarantined_counts = []
    for rd in data:
        qb = rd.get("quarantine_before", {})
        quarantined_counts.append(sum(1 for v in qb.values() if v > 0))

    fig, ax = plt.subplots()
    ax.fill_between(rounds, quarantined_counts, alpha=0.3, color='coral')
    ax.plot(rounds, quarantined_counts, marker='.', color='crimson', lw=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Clients in quarantine")
    ax.set_title(f"Clients in quarantine over rounds — {log['name']}")
    ax.grid(True, alpha=0.3)
    _safe_fig(f"quarantine_evolution_{log['name']}.png")


def plot_fedmad_dashboard(log):
    """Single dashboard with key FedMAD detection metrics."""
    data = log["data"]
    agent_names = data[0]["agent_names"]
    rounds = [r["round"] for r in data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"FedMAD Detection Dashboard — {log['name']}", fontsize=14, fontweight='bold')

    # ── top-left: flagged per agent ──
    ax = axes[0, 0]
    for aname in agent_names:
        flagged = [sum(1 for v in rd["per_agent_scores"].get(aname, {}).values() if v > 0.6) for rd in data]
        ax.plot(rounds, flagged, label=aname, marker='.', lw=1.2)
    removed = [len(rd.get("removed_clients", [])) for rd in data]
    ax.plot(rounds, removed, label='Removed', marker='x', lw=2, color='black', linestyle='--')
    ax.set_title("Clients flagged per round")
    ax.set_xlabel("Round")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── top-right: quarantine over rounds ──
    ax = axes[0, 1]
    qcount = [sum(1 for v in rd.get("quarantine_before", {}).values() if v > 0) for rd in data]
    ax.fill_between(rounds, qcount, alpha=0.3, color='coral')
    ax.plot(rounds, qcount, color='crimson', lw=1.5)
    ax.set_title("Clients in quarantine")
    ax.set_xlabel("Round")
    ax.grid(True, alpha=0.3)

    # ── bottom-left: avg score per agent ──
    ax = axes[1, 0]
    for aname in agent_names:
        avgs = [np.mean(list(rd["per_agent_scores"].get(aname, {}).values()) or [0]) for rd in data]
        ax.plot(rounds, avgs, label=aname, marker='.', lw=1.2)
    ax.set_title("Average anomaly score per agent")
    ax.set_xlabel("Round")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── bottom-right: malicious vs benign avg score ──
    ax = axes[1, 1]
    mal_scores = []
    ben_scores = []
    for rd in data:
        gt = {str(cid): int(m) for cid, m in zip(rd["client_ids"], rd["malicious_ground_truth"])}
        all_sc = []
        all_gt = []
        for aname in agent_names:
            scores = rd["per_agent_scores"].get(aname, {})
            for cid, sc in scores.items():
                all_sc.append(sc)
                all_gt.append(gt.get(cid, 0))
        mal_scores.append(np.mean([s for s, g in zip(all_sc, all_gt) if g]) if any(all_gt) else 0)
        ben_scores.append(np.mean([s for s, g in zip(all_sc, all_gt) if not g]) if any(not g for g in all_gt) else 0)
    ax.plot(rounds, mal_scores, label='Malicious avg', color='red', lw=1.5)
    ax.plot(rounds, ben_scores, label='Benign avg', color='green', lw=1.5)
    ax.axhline(0.6, color='gray', linestyle='--', lw=1, label='Threshold')
    ax.set_title("Avg score: malicious vs benign")
    ax.set_xlabel("Round")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _safe_fig(f"fedmad_dashboard_{log['name']}.png")


def plot_detection_effectiveness(log):
    """Per-agent precision, recall, F1 at threshold 0.6."""
    data = log["data"]
    agent_names = data[0]["agent_names"]

    results = {an: {"tp": 0, "fp": 0, "fn": 0} for an in agent_names}

    for rd in data:
        gt = {str(cid): int(m) for cid, m in zip(rd["client_ids"], rd["malicious_ground_truth"])}
        for aname in agent_names:
            scores = rd["per_agent_scores"].get(aname, {})
            for cid, sc in scores.items():
                is_mal = gt.get(cid, 0)
                flagged = sc > 0.6
                if is_mal and flagged:
                    results[aname]["tp"] += 1
                elif not is_mal and flagged:
                    results[aname]["fp"] += 1
                elif is_mal and not flagged:
                    results[aname]["fn"] += 1

    agents = []
    precisions = []
    recalls = []
    f1s = []
    for aname in agent_names:
        tp = results[aname]["tp"]
        fp = results[aname]["fp"]
        fn = results[aname]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        agents.append(aname)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    x = np.arange(len(agents))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, precisions, w, label='Precision', color='steelblue')
    ax.bar(x, recalls, w, label='Recall', color='mediumseagreen')
    ax.bar(x + w, f1s, w, label='F1-score', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Detection effectiveness (threshold=0.6) — {log['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i in range(len(agents)):
        for vals, offset in [(precisions, -w), (recalls, 0), (f1s, w)]:
            ax.text(i + offset, vals[i] + 0.02, f"{vals[i]:.2f}", ha='center', fontsize=7)
    _safe_fig(f"detection_effectiveness_{log['name']}.png")

    print(f"\nDetection effectiveness ({log['name']}):")
    print(f"{'Agent':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    for aname, p, r, f in zip(agents, precisions, recalls, f1s):
        print(f"{aname:<16} {p:>10.3f} {r:>10.3f} {f:>10.3f}")


# ──────────────────────── main ─────────────────────────────────

def main():
    print("=" * 55)
    print("FedMAD - plotting results")
    print("=" * 55)

    dataset = load_all_h5()
    if not dataset:
        print(f"No .h5 files found in {RESULTS_DIR}")
        return

    # ── load data ──
    for meta in dataset:
        meta.update(load_h5(meta['path']))

    print(f"\nFound {len(dataset)} .h5 file(s):")
    for d in dataset:
        print(f"  {d['path'].name}  ({len(d['rs_test_acc'])} rounds)")

    fprfrr_data = load_fpr_frr_csvs()
    print(f"\nFound {len(fprfrr_data)} FPR/FRR CSV file(s)")

    print("\n--- 1. Individual curves ---")
    plot_all_individuals(dataset)

    print("\n--- 2. Overlay by field ---")
    plot_individual_curve(dataset, 'rs_test_acc', 'Test Accuracy', 'Accuracy')
    plot_individual_curve(dataset, 'rs_train_loss', 'Train Loss', 'Loss')
    plot_individual_curve(dataset, 'rs_train_time', 'Time (s)', 'Time per Round')

    print("\n--- 3. Overlay grouped by dataset ---")
    plot_overlay_comparison(dataset)

    print("\n--- 4. Best accuracy bar chart ---")
    plot_best_acc_bar(dataset)

    print("\n--- 5. Total time bar ---")
    plot_total_time_bar(dataset)

    print("\n--- 6. FPR / FRR ---")
    plot_fpr_frr(fprfrr_data)

    print("\n--- 7. Summary dashboard ---")
    plot_summary_dashboard(dataset, fprfrr_data)

    print("\n--- 8. Rounds executed ---")
    plot_num_rounds(dataset)

    # ── FedMAD detection logs ──
    det_logs = load_detection_logs()
    if det_logs:
        print(f"\nFound {len(det_logs)} detection log(s):")
        for dl in det_logs:
            print(f"  {dl['path'].name}")
        print("\n--- 9. FedMAD: agent scores heatmap ---")
        for dl in det_logs:
            plot_agent_scores_heatmap(dl)
        print("\n--- 10. FedMAD: flagged per round ---")
        for dl in det_logs:
            plot_flagged_per_round(dl)
        print("\n--- 11. FedMAD: score distribution ---")
        for dl in det_logs:
            plot_score_distribution(dl)
        print("\n--- 12. FedMAD: agent agreement ---")
        for dl in det_logs:
            plot_agent_agreement_heatmap(dl)
        print("\n--- 13. FedMAD: quarantine evolution ---")
        for dl in det_logs:
            plot_quarantine_evolution(dl)
        print("\n--- 14. FedMAD: detection dashboard ---")
        for dl in det_logs:
            plot_fedmad_dashboard(dl)
        print("\n--- 15. FedMAD: detection effectiveness ---")
        for dl in det_logs:
            plot_detection_effectiveness(dl)
    else:
        print("\n(no detection_log JSON files found - run FedMAD training first)")

    print(f"\n{'=' * 55}")
    print(f"All plots saved in: {OUTPUT_DIR.resolve()}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()
