from flcore.madsystem.agents.agent_base import AgentBase


class AgentEmInspector(AgentBase):
    def __init__(self, args, encoder=None, **kwargs):
        super().__init__(args, name="EmInspector")
        self.encoder = encoder

    def analyze(self, client_models, global_model, metadata):
        n = len(client_models)
        return [0.5] * n
