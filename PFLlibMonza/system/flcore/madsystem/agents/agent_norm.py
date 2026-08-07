"""
AgentNormDefense — Defesa classica contra envenenamento de modelo por NORMA.

Calcula a distancia (norma Lp) entre o modelo enviado pelo cliente e o
modelo global:  ||W_i - W_global||_p  para p = 2 (L2), p = 3 (L3),
p = float('inf') (L∞).

Um cliente malicioso envia parametros muito divergentes do modelo global
(ex.: inversao de sinais, ruido gaussiano, pesos aleatorios), o que produz
uma norma de desvio anormalmente alta em relacao aos clientes benignos.

Score = robust_anomaly_scores(normas) — z-score robusto (mediana/MAD) + sigmoide,
        sem forcar remocoes em rounds sem ataques.
"""

import torch
from flcore.madsystem.agents.agent_base import AgentBase


class AgentNormDefense(AgentBase):
    def __init__(self, args, p=2, name=None):
        self.p = p
        super().__init__(args, name=name or f"L{p if p != float('inf') else 'inf'}Norm")

    def analyze(self, client_models, global_model, metadata=None):
        """
        Para cada modelo de cliente, calcula ||W_i - W_global||_p.
        Retorna lista de scores de anomalia (1 = muito suspeito).
        """
        global_dict = global_model.state_dict()
        norms = []
        for model in client_models:
            diff = [
                (p - global_dict[name]).flatten()
                for name, p in model.state_dict().items()
                if name in global_dict
            ]
            if not diff:
                norms.append(0.0)
                continue
            flat = torch.cat(diff).to(self.device)
            if self.p == float('inf'):
                norms.append(flat.abs().max().item())
            else:
                norms.append(torch.norm(flat, p=self.p).item())

        return self.robust_anomaly_scores(norms)
