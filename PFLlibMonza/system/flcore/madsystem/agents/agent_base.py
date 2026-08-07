from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class AgentBase(ABC):
    def __init__(self, args, name="AgentBase"):
        self.args = args
        self.name = name
        self.device = args.device if hasattr(args, "device") else "cpu"
        self.verbose = args.verbose if hasattr(args, "verbose") else False

    @abstractmethod
    def analyze(self, client_models, global_model, metadata):
        ...
    
    def get_name(self):
        return self.name
    
    def normalize_scores(self, raw_scores):
        scores = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)
        if scores.numel() == 0:
            return []
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return [0.5] * len(raw_scores)
        return ((scores - min_s) / (max_s - min_s)).tolist()

    def robust_anomaly_scores(self, raw_scores):
        """
        Converte valores brutos em scores de anomalia em [0,1) usando
        z-score robusto (mediana + MAD) com sigmoide.

        - Sem outliers: todos os scores ficam proximos de 0.5 (sem remocoes)
        - Com outlier (ex.: envenenamento): o desvio vira score alto
        """
        values = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)
        if values.numel() == 0:
            return []
        if values.numel() == 1:
            return [0.5]
        median = values.median()
        mad = (values - median).abs().median() + 1e-8
        z = (values - median) / (1.4826 * mad)
        return torch.sigmoid(z).tolist()
    
    def cosine_similarity(self, a, b):
        a_f = torch.cat([p.flatten().to(self.device) for p in a.values()])
        b_f = torch.cat([p.flatten().to(self.device) for p in b.values()])
        return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()

    def euclidean_distance(self, a, b):
        a_f = torch.cat([p.flatten().to(self.device) for p in a.values()])
        b_f = torch.cat([p.flatten().to(self.device) for p in b.values()])
        return torch.norm(a_f - b_f).item()

    def flatten_params(self, model_dict):
        return torch.cat([p.flatten().to(self.device) for p in model_dict.values()])