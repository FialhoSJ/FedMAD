import torch
import torch.nn.functional as F
from flcore.madsystem.agents.agent_base import AgentBase


class AgentBehavior(AgentBase):
    def __init__(self, args, **kwargs):
        super().__init__(args, name="Behavior")
        self.lookback = getattr(args, 'bhv_lookback', 5)
        self.client_history = {}

    def analyze(self, client_models, global_model, metadata):
        client_ids = metadata.get("client_ids", [])
        n = len(client_models)
        if n < 1:
            return [0.5] * n

        client_models = [m.to(self.device) for m in client_models]
        global_model = global_model.to(self.device)

        scores = []
        for i, cid in enumerate(client_ids):
            current_sd = client_models[i].state_dict()
            prev = self.client_history.get(cid, [])

            if len(prev) == 0:
                sim = self.cosine_similarity(current_sd, global_model.state_dict())
            else:
                sim = self.cosine_similarity(current_sd, prev[-1])

            scores.append(1 - sim)

            if cid not in self.client_history:
                self.client_history[cid] = []
            self.client_history[cid].append(
                {k: v.detach().cpu() for k, v in current_sd.items()}
            )
            if len(self.client_history[cid]) > self.lookback:
                self.client_history[cid].pop(0)

        return self.normalize_scores(scores)
