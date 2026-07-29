import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientavg import clientAVG


class ClientMAD(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.encoder = None
        self.ssl_epochs = args.ssl_epochs if hasattr(args, 'ssl_epochs') else 0
        ssl_pd = args.ssl_proj_dim if hasattr(args, 'ssl_proj_dim') else 128
        if ssl_pd > 0:
            enc_dim = 1600 if "Cifar10" in args.dataset else 1024
            self.projection = nn.Sequential(
                nn.Linear(enc_dim, enc_dim // 2),
                nn.ReLU(),
                nn.Linear(enc_dim // 2, ssl_pd),
            ).to(self.device)

    def set_encoder(self, encoder):
        self.encoder = copy.deepcopy(encoder)

    def ssl_train(self):
        if self.encoder is None or self.ssl_epochs == 0:
            return
        self.encoder.train()
        self.projection.train()
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.projection.parameters()),
            lr=self.learning_rate,
        )
        for _ in range(self.ssl_epochs):
            for x, _ in trainloader:
                if type(x) == type([]):
                    x = x[0]
                x = x.to(self.device)
                # SimCLR: duas aumentações da mesma imagem
                x_aug1, x_aug2 = x, x  # placeholder: aplicar transformações
                h1 = self.projection(self.encoder(x_aug1))
                h2 = self.projection(self.encoder(x_aug2))
                # NT-Xent loss
                h1 = nn.functional.normalize(h1, dim=1)
                h2 = nn.functional.normalize(h2, dim=1)
                logits = torch.mm(h1, h2.T) / 0.5
                labels = torch.arange(x.size(0)).to(self.device)
                loss = nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train(self):
        self.ssl_train()
        super().train()

    def send_local_model(self, round):
        return self.model