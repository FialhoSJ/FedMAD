import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "PFLlibMonza/results"


def load_agent_log(filepath):
    with open(filepath) as f:
        return json.load(f)


def plot_all(data, save_path=None):
    agent_names = data["agent_names"]
    n_agents = len(agent_names)
    rounds = [e["round"] for e in data["agent_round_log"]]
    n_rounds = len(rounds)
    n_clients = data["num_clients"]
    malicious_mask = np.array(data["malicious_mask"])
    malicious_indices = data["malicious_indices"]

    fig, axes = plt.subplots(n_agents + 1, 1, figsize=(14, 3 * (n_agents + 1)),
                             sharex=True)

    # ---- Matriz de scores por agente ----
    for ax_idx, agent_name in enumerate(agent_names):
        ax = axes[ax_idx]
        score_matrix = np.full((n_rounds, n_clients), np.nan)
        for r_idx, entry in enumerate(data["agent_round_log"]):
            cids = entry["client_ids_uploaded"]
            scores = entry["agent_scores"].get(agent_name, {})
            for cid_str, score in scores.items():
                cid = int(cid_str)
                if cid < n_clients:
                    score_matrix[r_idx, cid] = score

        im = ax.imshow(score_matrix.T, aspect="auto", cmap="Reds",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(n_clients))
        ax.set_ylabel("Client")
        ax.set_title(f"{agent_name} - Anomaly Scores")
        cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cb.set_label("Score")

        # Destacar clientes maliciosos
        for mi in malicious_indices:
            ax.axhline(y=mi - 0.5, color="blue", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(y=mi + 0.5, color="blue", linewidth=0.8, linestyle="--", alpha=0.5)

    # ---- Final scores agregados ----
    ax = axes[-1]
    final_matrix = np.full((n_rounds, n_clients), np.nan)
    for r_idx, entry in enumerate(data["agent_round_log"]):
        for cid_str, score in entry["final_scores"].items():
            cid = int(cid_str)
            if cid < n_clients:
                final_matrix[r_idx, cid] = score

    im = ax.imshow(final_matrix.T, aspect="auto", cmap="Reds",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n_clients))
    ax.set_ylabel("Client")
    ax.set_xlabel("Round")
    ax.set_title("Final Aggregated Score")
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cb.set_label("Score")

    for mi in malicious_indices:
        ax.axhline(y=mi - 0.5, color="blue", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(y=mi + 0.5, color="blue", linewidth=0.8, linestyle="--", alpha=0.5)

    # Linha azul tracejada = clientes maliciosos (ground truth)
    fig.text(0.02, 0.5, "--- blue = malicious (ground truth)", va="center",
             rotation=90, fontsize=9, color="blue", alpha=0.6)

    fig.suptitle(f"MAD Detection - {data['n_client_malicious']}/{data['num_clients']} malicious, {data['global_rounds']} rounds",
                 fontsize=14)
    plt.tight_layout(rect=[0.04, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_scores_over_rounds(data, save_path=None):
    agent_names = data["agent_names"]
    n_clients = data["num_clients"]
    rounds = [e["round"] for e in data["agent_round_log"]]

    fig, axes = plt.subplots(n_clients, 1, figsize=(12, 2.5 * n_clients),
                             sharex=True, squeeze=False)

    for cid in range(n_clients):
        ax = axes[cid, 0]
        cid_str = str(cid)

        rounds_plot = []
        scores_by_agent = {name: [] for name in agent_names}
        final_scores = []

        for entry in data["agent_round_log"]:
            if cid_str in entry.get("final_scores", {}):
                rounds_plot.append(entry["round"])
                for name in agent_names:
                    s = entry["agent_scores"].get(name, {}).get(cid_str, np.nan)
                    scores_by_agent[name].append(s)
                final_scores.append(entry["final_scores"][cid_str])

        if not rounds_plot:
            continue

        for name in agent_names:
            ax.plot(rounds_plot, scores_by_agent[name], "o-", label=name, alpha=0.7)
        ax.plot(rounds_plot, final_scores, "k-", linewidth=2, label="Final", alpha=0.9)

        is_mal = "MALICIOUS" if cid in data["malicious_indices"] else "benign"
        ax.set_title(f"Client {cid} ({is_mal})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    ax.set_xlabel("Round")
    fig.suptitle("Anomaly Scores per Client per Round", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_agent_results.py <agent_log.json> [--per-client]")
        print("\nGlob: python plot_agent_results.py PFLlibMonza/results/*_agentlog.json")
        sys.exit(1)

    path = sys.argv[1]
    per_client = "--per-client" in sys.argv

    if "*" in path or "?" in path:
        import glob
        files = glob.glob(path)
    elif os.path.isfile(path):
        files = [path]
    else:
        files = []

    if not files:
        print(f"No files found: {path}")
        sys.exit(1)

    for fp in files:
        print(f"\nLoading: {fp}")
        data = load_agent_log(fp)
        base = os.path.splitext(fp)[0]
        if per_client:
            plot_scores_over_rounds(data, save_path=f"{base}_per_client.png")
        else:
            plot_all(data, save_path=f"{base}_heatmap.png")
            plot_scores_over_rounds(data, save_path=f"{base}_per_client.png")
