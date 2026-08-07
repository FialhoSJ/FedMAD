"""
AgentNormDefense — Defesa classica contra envenenamento de modelo por NORMA.

Calcula a distancia (norma Lp) entre o modelo enviado pelo cliente e o
modelo global:  ||W_i - W_global||_p, para p = 2 (L2) e p = 3 (L3).

As normas L2 e L3 sao sinais quase redundantes (mesma familia de desvio) e
por isso sao combinadas num UNICO agent "Norm": o score final de cada
cliente e a media dos scores robustos de L2 e L3.

Um cliente malicioso envia parametros muito divergentes do modelo global
(ex.: inversao de sinais, ruido numerico, pesos aleatorios), o que produz
uma norma de desvio anormalmente alta em relacao aos clientes benignos.

Score = robust_anomaly_scores(normas) — z-score robusto (mediana/MAD) + sigmoide,
        sem forcar remocoes em rounds sem ataques.
"""

import torch
from flcore.madsystem.agents.agent_base import AgentBase


class AgentNormDefense(AgentBase):
    # Ps a combinar (L2 e L3 por padrao).
    def __init__(self, args, name="Norm", ps=(2, 3)):
        self.ps = ps
        super().__init__(args, name=name)

    def _norm_p(self, flat, p):
        """Calcula a norma Lp de um vetor para p = 2, 3 ou float('inf') (L∞)."""
        if p == float('inf'):
            return flat.abs().max().item()
        return torch.norm(flat, p=p).item()

    def analyze(self, client_models, global_model, metadata=None):
        """
        Para cada modelo de cliente, calcula a norma de desvio para cada p
        (L2 e L3) e combina as scores robustas na media por cliente.
        Retorna lista de scores de anomalia (1 = muito suspeito).
        """
        global_dict = global_model.state_dict()
        raw_by_p = {p: [] for p in self.ps}

        for model in client_models:
            diff = []
            for name, param in model.state_dict().items():
                if name in global_dict:
                    diff.append((param - global_dict[name]).flatten())
            if not diff:
                for p in self.ps:
                    raw_by_p[p].append(0.0)
                continue
            flat = torch.cat(diff).to(self.device)
            for p in self.ps:
                raw_by_p[p].append(self._norm_p(flat, p))

        # Converte cada norma em score robusto e combina (media) por cliente
        scores_by_p = {p: self.robust_anomaly_scores(v) for p, v in raw_by_p.items()}
        n = len(client_models)
        if n == 0:
            return []
        return [sum(scores_by_p[p][i] for p in self.ps) / len(self.ps) for i in range(n)]