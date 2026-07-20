import torch
from flcore.madsystem.agents.agent_base import AgentBase


class AgentEmInspector(AgentBase):
    def __init__(self, args, encoder=None, **kwargs):
        super().__init__(args, name="EmInspector")
        self.encoder = encoder

    def analyze(self, client_models, global_model, metadata):
        n = len(client_models)
        if n < 2:
            return [0.5] * n

        flat = torch.stack([
            self.flatten_params(m.state_dict()) for m in client_models
        ])

        norms = torch.norm(flat, dim=1, keepdim=True)
        normalized = flat / (norms + 1e-10)

        sim_matrix = normalized @ normalized.T
        avg_sim = (sim_matrix.sum(dim=1) - 1) / (n - 1)

        scores = 1 - avg_sim
        return self.normalize_scores(scores.tolist())
