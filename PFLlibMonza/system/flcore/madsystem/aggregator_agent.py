"""
AggregatorAgent base - média aritmética dos scores dos 3 agentes.
Mantido como fallback quando SLM não está disponível.
"""

class AggregatorAgent:
    def __init__(self, args):
        self.args = args

    def aggregate(self, client_scores, metadata=None):
        result = {}
        for cid, scores in client_scores.items():
            result[cid] = sum(scores) / len(scores)
        return result
