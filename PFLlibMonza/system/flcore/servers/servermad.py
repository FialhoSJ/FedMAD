import time
import copy
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
                for agent in self.agents:
                    scores = agent.analyze(
                        self.uploaded_models, self.global_model, metadata
                    )
                    for cid, sc in zip(self.uploaded_ids, scores):
                        client_scores[cid].append(sc)

                # 6. Agregador (SLM ou media aritmetica) combina os scores
                final_scores = self.aggregator.aggregate(client_scores, metadata=metadata)

                # 7. Remove modelos de clientes com score > 0.6 (threshold)
                for idx in range(len(self.uploaded_ids) - 1, -1, -1):
                    cid = self.uploaded_ids[idx]
                    if final_scores.get(cid, 0) > 0.6:
                        print(f"Removing client {cid} (score={final_scores[cid]:.4f})")
                        self.set_client_quarantine(cid)
                        del self.uploaded_models[idx]
                        del self.ids[idx]
                        del self.uploaded_ids[idx]
                        del self.uploaded_weights[idx]

                # Re-normaliza os pesos apos remocao
                self.uploaded_weights = [
                    w / sum(self.uploaded_weights) for w in self.uploaded_weights
                ]

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

            # Guarda historico de modelos para detecao temporal
            self.history.append(
                [copy.deepcopy(m.state_dict()) for m in self.uploaded_models]
            )

        # Resultados finais
        print("\nBest accuracy:", max(self.rs_test_acc))
        print("Average time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()