from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class AgentBase(ABC):
    def __init__(self, args, name="AgentBase"):
        self.args = args
        self.name = name
        self.verbose = args.verbose if hasattr(args, "verbose") else False

    @abstractmethod
    def analyze(self, client_models, global_model, metadata):
        ...
    
    def get_name(self):
        return self.name
    
    def normalize_scores(self, raw_scores):
        scores = torch.tensor(raw_scores, dtype=torch.float32)
        if scores.numel() ==0:
            return []
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return [0.5] * len(raw_scores)
        return ((scores - min_s) / (max_s - min_s)).tolist()
    
    @staticmethod
    def cosine_similarity(a, b):
        a_f = torch.cat([p.flatten() for p in a.values()])
        b_f = torch.cat([p.flatten() for p in b.values()])
        return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()

    @staticmethod
    def euclidean_distance(a, b):
        a_f = torch.cat([p.flatten() for p in a.values()])
        b_f = torch.cat([p.flatten() for p in b.values()])
        return torch.norm(a_f - b_f).item()

    @staticmethod
    def flatten_params(model_dict):
        return torch.cat([p.flatten() for p in model_dict.values()])