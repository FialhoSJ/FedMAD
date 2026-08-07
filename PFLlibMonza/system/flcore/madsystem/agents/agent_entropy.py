"""
AgentEntropyDefense — Defesa classica contra envenenamento por ENTROPIA DE SHANNON.

Calcula a entropia de Shannon da distribuicao de probabilidade formada
pelos valores absolutos dos parametros do modelo enviado por cada cliente:

    H = -Σ p_i * log(p_i),  p_i = |w_i| / Σ|w_j|

Modelos bem treinados (benignos) concentram massa em poucos parametros
(entropia relativamente baixa e estavel entre clientes). Modelos envenenados
— pesos aleatorios, ruido gaussiano, zero-ing — tendem a apresentar
entropia fora do padrao dos demais.

Score = robust_anomaly_scores(H) — entropia destoante vira score alto.
"""

import torch
from flcore.madsystem.agents.agent_base import AgentBase


class AgentEntropyDefense(AgentBase):
    def __init__(self, args):
        super().__init__(args, name="Entropy")

    @staticmethod
    def _shannon_entropy(flat_tensor):
        """Entropia de Shannon dos valores absolutos normalizados como distribuicao."""
        w = flat_tensor.abs()
        w = w + 1e-12  # evita log(0)
        p = w / w.sum()
        return -(p * p.log()).sum().item()

    def analyze(self, client_models, global_model, metadata=None):
        """
        Para cada modelo de cliente, calcula a entropia de Shannon dos parametros.
        Retorna lista de scores de anomalia (1 = muito suspeito).
        """
        entropies = []
        for model in client_models:
            flat = self.flatten_params(model.state_dict())
            entropies.append(self._shannon_entropy(flat))

        return self.robust_anomaly_scores(entropies)
