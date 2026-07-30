import torch
import torch.nn as nn
from flcore.madsystem.agents.agent_base import AgentBase


class _Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        h = min(input_dim, 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(),
            nn.Linear(h, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AgentFedREDefense(AgentBase):
    def __init__(self, args, **kwargs):
        super().__init__(args, name="FedREDefense")
        self.latent_dim = getattr(args, 'fedre_latent', 16)
        self.epochs = getattr(args, 'fedre_epochs', 10)
        self.lr = getattr(args, 'fedre_lr', 0.01)
        self.autoencoder = None

    def _extract_features(self, model_dict):
        feats = []
        for v in model_dict.values():
            p = v.flatten().to(self.device)
            feats.extend([p.mean().item(), p.std().item(), p.norm().item()])
        return torch.tensor(feats, device=self.device)

    def analyze(self, client_models, global_model, metadata):
        client_models = [m.to(self.device) for m in client_models]

        features = torch.stack([
            self._extract_features(m.state_dict()) for m in client_models
        ])

        n, d = features.shape
        if n < 3:
            return [0.5] * n

        if self.autoencoder is None or self.autoencoder.encoder[0].in_features != d:
            self.autoencoder = _Autoencoder(d, self.latent_dim).to(self.device)

        ae = self.autoencoder
        opt = torch.optim.Adam(ae.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            recon = ae(features)
            loss = nn.MSELoss()(recon, features)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            recon = ae(features)
            errors = (features - recon).norm(dim=1)

        return self.normalize_scores(errors.tolist())
