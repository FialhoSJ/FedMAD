class AggregatorAgent:
    def __init__(self, args):
        self.args = args

    def aggregate(self, client_scores):
        result = {}
        for cid, scores in client_scores.items():
            result[cid] = sum(scores) / len(scores)
        return result
