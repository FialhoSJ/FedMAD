"""
AgentCosineDefense — Defesa classica contra envenenamento por SIMILARIDADE DE COSSENO.

Calcula a similaridade de cosseno entre os parametros de cada modelo de
cliente e o modelo global:  sim(W_i, W_global) em [-1, 1].

Clientes benignos treinam a partir do mesmo modelo global e permanecem
proximos (sim alta); clientes maliciosos (label flip, inversao de sinais,
ruido) divergem em direcao (sim baixa ou negativa).

Score = robust_anomaly_scores(1 - sim) — sim baixa vira score alto.
"""

import torch
from flcore.madsystem.agents.agent_base import AgentBase


class AgentCosineDefense(AgentBase):
    def __init__(self, args):
        super().__init__(args, name="Cosine")

    def analyze(self, client_models, global_model, metadata=None):
        """
        Para cada modelo de cliente, calcula 1 - cosine(W_i, W_global).
        Retorna lista de scores de anomalia (1 = muito suspeito).
        """
        global_flat = self.flatten_params(global_model.state_dict())
        if global_flat.norm() < 1e-8:
            return [0.5] * len(client_models)

        raw = []
        for model in client_models:
            client_flat = self.flatten_params(model.state_dict())
            sim = torch.nn.functional.cosine_similarity(
                client_flat.unsqueeze(0), global_flat.unsqueeze(0)
            ).item()
            raw.append(1.0 - sim)  # 0 = identico, 2 = oposto

        return self.robust_anomaly_scores(raw)
