import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from threading import Thread

from flcore.servers.serverbase import Server
from flcore.clients.clientmad import ClientMAD
from flcore.trainmodel.models import FedAvgCNN


class ServerMAD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(ClientMAD)

        # Codificador partilhado (extrai representacoes intermédias das CNNs)
        self.encoder = self._build_encoder(args)
        self.history = []  # historico de modelos enviados (para detecao temporal)

        # --- Tracking para graficos ---
        self.agent_round_log = []       # lista de dicts por round

        # Importacoes tardias para evitar dependencias circulares
        from flcore.madsystem.agents.agent_emInspector import AgentEmInspector
        from flcore.madsystem.agents.agent_fedREDefense import AgentFedREDefense
        from flcore.madsystem.agents.agent_behavior import AgentBehavior
        from flcore.madsystem.agents.agent_fedllmguard import AgentFedLLMGuard
        from flcore.madsystem.agents.agent_slm_aggregator import SLMAggregatorAgent
        from flcore.madsystem.aggregator_agent import AggregatorAgent

        # Detectores especializados (cada um produz scores de anomalia)
        self.agents = []
        self.agent_names = []
        if getattr(args, 'agent_em', True):
            self.agents.append(AgentEmInspector(args, encoder=self.encoder))
            self.agent_names.append("EmInspector")
        if getattr(args, 'agent_fedre', True):
            self.agents.append(AgentFedREDefense(args))
            self.agent_names.append("FedREDefense")
        if getattr(args, 'agent_bhv', True):
            self.agents.append(AgentBehavior(args))
            self.agent_names.append("Behavior")
        if getattr(args, 'agent_flg', True):
            self.agents.append(AgentFedLLMGuard(args))
            self.agent_names.append("FedLLMGuard")
        print(f"[MAD] Agentes ativos ({len(self.agents)}): {', '.join(self.agent_names)}")
        # Agregador: substitui a media aritmetica por raciocinio com SLM
        # (Phi-3-mini ou TinyLlama) que analisa os scores + metadados
        if getattr(args, 'slm_enabled', True):
            self.aggregator = SLMAggregatorAgent(args)      # decisao com LLM
        else:
            self.aggregator = AggregatorAgent(args)          # fallback: media aritmetica

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def _build_encoder(self, args):
        """Constroi o codificador CNN partilhado (FedAvgCNN sem a cabeça de classificacao)."""
        in_f = 3 if "Cifar10" in args.dataset else 1
        dim = 1600 if "Cifar10" in args.dataset else 1024
        encoder = FedAvgCNN(in_features=in_f, num_classes=args.num_classes, dim=dim)
        encoder.fc = nn.Identity()  # remove a camada fully-connected final
        return encoder.to(args.device)

    def send_models(self):
        """Envia o modelo global + codificador para todos os clientes."""
        super().send_models()
        for client in self.clients:
            if hasattr(client, 'set_encoder'):
                client.set_encoder(self.encoder)

    def set_client_quarantine(self, client_id):
        """Marca um cliente como suspeito: incrementa o contador de quarentenas."""
        self.client_quarantine_dict[client_id]['quarentena'] += 1
        # O tempo de quarentena aumenta progressivamente (2x o numero de detecoes)
        self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['quarentena'] * 2

    def decrease_quarentine(self, client_id):
        """Reduz o tempo restante de quarentena em 1 round."""
        if self.client_quarantine_dict[client_id]['roundsQuarent'] > 0:
            self.client_quarantine_dict[client_id]['roundsQuarent'] -= 1

    def train(self):
        """Loop principal de treino federado com detecao de clientes maliciosos."""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # 1. Reduz a quarentena de todos os clientes
            for j in range(self.num_clients):
                self.decrease_quarentine(j)

            # 2. Treino local dos clientes (em paralelo)
            threads = [Thread(target=client.train) for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            # 3. Recolhe os modelos enviados pelos clientes
            self.receive_models()

            # 4. Executa detecao de anomalias (apenas apos o round 0)
            if i > 0 and len(self.uploaded_models) > 0:
                # IDs dos clientes atualmente em quarentena (metadados para o SLM)
                quarantined_ids = [
                    cid for cid, status in self.client_quarantine_dict.items()
                    if status['roundsQuarent'] > 0
                ]

                # Metadados para os agentes e para o SLM
                metadata = {
                    "round": i,
                    "client_ids": self.uploaded_ids,
                    "history": self.history,
                    "encoder": self.encoder,
                    "quarantined_ids": quarantined_ids,
                    "agent_names": self.agent_names,
                }

                # 5. Cada agente especializado produz scores de anomalia
                client_scores = {cid: [] for cid in self.uploaded_ids}
                agent_scores_by_name = {}
                agent_times = {}
                for agent in self.agents:
                    t0 = time.time()
                    scores = agent.analyze(
                        self.uploaded_models, self.global_model, metadata
                    )
                    agent_times[agent.name] = time.time() - t0
                    agent_scores_by_name[agent.name] = scores
                    for cid, sc in zip(self.uploaded_ids, scores):
                        client_scores[cid].append(sc)

                # Embeddings compactos dos modelos enviados (para t-SNE)
                client_embeddings = {
                    cid: self._pooled_embedding(m)
                    for cid, m in zip(self.uploaded_ids, self.uploaded_models)
                }
                global_embedding = self._pooled_embedding(self.global_model)

                # 6. Agregador (SLM ou media aritmetica) combina os scores
                t0 = time.time()
                final_scores = self.aggregator.aggregate(client_scores, metadata=metadata)
                agg_time = time.time() - t0

                # 7. Remove modelos de clientes com score > 0.6 (threshold)
                removed_ids = []
                for idx in range(len(self.uploaded_ids) - 1, -1, -1):
                    cid = self.uploaded_ids[idx]
                    if final_scores.get(cid, 0) > 0.6:
                        print(f"Removing client {cid} (score={final_scores[cid]:.4f})")
                        removed_ids.append(cid)
                        self.set_client_quarantine(cid)
                        del self.uploaded_models[idx]
                        del self.ids[idx]
                        del self.uploaded_ids[idx]
                        del self.uploaded_weights[idx]

                # Re-normaliza os pesos apos remocao
                self.uploaded_weights = [
                    w / sum(self.uploaded_weights) for w in self.uploaded_weights
                ]

                # --- Log dos agentes para grafico ---
                log_entry = {
                    "round": i,
                    "client_ids_uploaded": list(self.uploaded_ids),
                    "agent_scores": {
                        name: dict(zip(self.uploaded_ids, scores))
                        for name, scores in agent_scores_by_name.items()
                    },
                    "final_scores": dict(final_scores),
                    "removed_ids": removed_ids,
                    "quarantined_ids": quarantined_ids,
                    "agent_times": agent_times,
                    "agg_time": agg_time,
                    "client_embeddings": client_embeddings,
                    "global_embedding": global_embedding,
                }
                self.agent_round_log.append(log_entry)

            # 8. Agregacao dos modelos (FedAvg)
            self.aggregate_parameters()

            # Logs de desempenho
            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            # Paragem automatica se a acuracia estabilizar
            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

            # Guarda historico RECENTE de modelos para detecao temporal
            # (limitado ao lookback e mantido em CPU para nao estourar a VRAM)
            max_history = getattr(self.args, 'bhv_lookback', 5)
            self.history.append(
                [
                    {k: v.detach().cpu() for k, v in m.state_dict().items()}
                    for m in self.uploaded_models
                ]
            )
            if len(self.history) > max_history:
                self.history.pop(0)

        # Resultados finais
        print("\nBest accuracy:", max(self.rs_test_acc))
        print("Average time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_agent_results()
        self.save_global_model()

    def _pooled_embedding(self, model, target_dim=512):
        """Embedding compacto dos parametros de um modelo para o t-SNE:
        flatten + normalizacao L2 + mean-pool por janelas (CPU-safe)."""
        flat = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
        flat = flat / (flat.norm() + 1e-10)
        n = flat.numel()
        w = max(1, n // target_dim)
        pad = (-n) % w
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        pooled = flat.view(-1, w).mean(dim=1)
        return pooled.tolist()

    def save_agent_results(self):
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        algo = f"{self.dataset}_{self.algorithm}_{self.cc}_{int(self.rate_client_fake*100)}_{self.n_client_malicious}_{self.args.atack}"
        algo = f"{algo}_{self.goal}_{self.times}"
        file_path = result_path + f"{algo}_agentlog.json"

        # Ground truth: mascara binaria dos indices maliciosos
        malicious_mask = [1 if i in self.index_malicious else 0 for i in range(self.num_clients)]

        # Converte agent_round_log para serializavel (JSON-safe)
        log_serializable = []
        for entry in self.agent_round_log:
            agent_scores_clean = {}
            for name, scores_dict in entry["agent_scores"].items():
                agent_scores_clean[name] = {str(k): float(v) for k, v in scores_dict.items()}
            entry_clean = {
                "round": int(entry["round"]),
                "client_ids_uploaded": [int(c) for c in entry["client_ids_uploaded"]],
                "agent_scores": agent_scores_clean,
                "final_scores": {str(k): float(v) for k, v in entry["final_scores"].items()},
                "removed_ids": [int(c) for c in entry["removed_ids"]],
                "quarantined_ids": [int(c) for c in entry["quarantined_ids"]],
                "agent_times": {k: float(v) for k, v in entry.get("agent_times", {}).items()},
                "agg_time": float(entry.get("agg_time", 0.0)),
                "client_embeddings": {
                    str(k): [float(x) for x in v] for k, v in entry.get("client_embeddings", {}).items()
                },
                "global_embedding": [float(x) for x in entry.get("global_embedding", [])],
            }
            log_serializable.append(entry_clean)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return super().default(obj)

        data = {
            "num_clients": int(self.num_clients),
            "n_client_malicious": int(self.n_client_malicious),
            "global_rounds": int(self.global_rounds),
            "atack": str(self.args.atack),
            "rate_client_fake": float(self.rate_client_fake),
            "agent_names": self.agent_names,
            "malicious_indices": [int(i) for i in self.index_malicious],
            "malicious_mask": malicious_mask,
            "agent_round_log": log_serializable,
            "rs_test_acc": [float(x) for x in self.rs_test_acc],
            "rs_test_auc": [float(x) for x in self.rs_test_auc],
            "rs_asr": [float(x) for x in self.rs_asr],
            "rs_ba": [float(x) for x in self.rs_ba],
            "budget": [float(x) for x in self.Budget],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        print(f"[MAD] Agent results saved to {file_path}")