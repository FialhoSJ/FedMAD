from flcore.madsystem.agents.agent_base import AgentBase


class AgentBehavior(AgentBase):
    def __init__(self, args, **kwargs):
        super().__init__(args, name="Behavior")

    def analyze(self, client_models, global_model, metadata):
        n = len(client_models)
        return [0.5] * n
