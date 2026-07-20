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

        self.encoder = self._build_encoder(args)
        self.history = []

        from flcore.madsystem.agents.agent_emInspector import AgentEmInspector
        from flcore.madsystem.agents.agent_fedREDefense import AgentFedREDefense
        from flcore.madsystem.agents.agent_behavior import AgentBehavior
        from flcore.madsystem.aggregator_agent import AggregatorAgent

        self.agents = [
            AgentEmInspector(args, encoder=self.encoder),
            AgentFedREDefense(args),
            AgentBehavior(args),
        ]
        self.aggregator = AggregatorAgent(args)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def _build_encoder(self, args):
        in_f = 3 if "Cifar10" in args.dataset else 1
        dim = 1600 if "Cifar10" in args.dataset else 1024
        encoder = FedAvgCNN(in_features=in_f, num_classes=args.num_classes, dim=dim)
        encoder.fc = nn.Identity()
        return encoder.to(args.device)

    def send_models(self):
        super().send_models()
        for client in self.clients:
            if hasattr(client, 'set_encoder'):
                client.set_encoder(self.encoder)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for j in range(self.num_clients):
                self.decrease_quarentine(j)

            threads = [Thread(target=client.train) for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()

            if i > 0 and len(self.uploaded_models) > 0:
                metadata = {
                    "round": i,
                    "client_ids": self.uploaded_ids,
                    "history": self.history,
                    "encoder": self.encoder,
                }

                client_scores = {cid: [] for cid in self.uploaded_ids}
                for agent in self.agents:
                    scores = agent.analyze(
                        self.uploaded_models, self.global_model, metadata
                    )
                    for cid, sc in zip(self.uploaded_ids, scores):
                        client_scores[cid].append(sc)

                final_scores = self.aggregator.aggregate(client_scores)

                for idx in range(len(self.uploaded_ids) - 1, -1, -1):
                    cid = self.uploaded_ids[idx]
                    if final_scores.get(cid, 0) > 0.6:
                        print(f"Removing client {cid} (score={final_scores[cid]:.4f})")
                        self.set_client_quarantine(cid)
                        del self.uploaded_models[idx]
                        del self.ids[idx]
                        del self.uploaded_ids[idx]
                        del self.uploaded_weights[idx]

                self.uploaded_weights = [
                    w / sum(self.uploaded_weights) for w in self.uploaded_weights
                ]

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

            self.history.append(
                [copy.deepcopy(m.state_dict()) for m in self.uploaded_models]
            )

        print("\nBest accuracy:", max(self.rs_test_acc))
        print("Average time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()