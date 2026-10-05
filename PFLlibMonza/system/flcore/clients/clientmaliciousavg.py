import numpy as np
import time
import random
import torch
from flcore.clients.clientavg import clientAVG

#from utils.privacy import *

from flcore.attack.attack import *


class ClientMaliciousAVG(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.rate_client_fake = args.rate_client_fake
        self.atack = args.atack

        self.latest_global_model = None
        self.attack_scale = float(getattr(args, "attack_scale", 10.0))
        self.attack_noise_snr = float(getattr(args, "attack_noise_snr", 1.0))
        #self.delay_atk = args.delay_atk
        self.round_init_atk = args.round_init_atk

    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def set_parameters(self, model):
        self.latest_global_model = model
        return super().set_parameters(model)

    def _train_with_label_flip(self):
        """Train on the client's images after cyclically shifting every label."""
        trainloader = self.load_train_data()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = ((y.to(self.device) + 1) % self.num_classes)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                loss = self.loss(self.model(x), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model

    def _train_with_targeted_label_flip(self, source_label, target_label):
        """Train after changing only the selected source class to its target."""
        source_label = int(source_label)
        target_label = int(target_label)
        if source_label == target_label:
            raise ValueError("source_label and target_label must differ")

        trainloader = self.load_train_data()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                labels = y.to(self.device)
                labels = torch.where(
                    labels == source_label,
                    torch.full_like(labels, target_label),
                    labels,
                )
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                loss = self.loss(self.model(x), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model
    
    def send_local_model(self, round):
        if round <= self.round_init_atk:
            return self.model
        self.is_malicious = np.random.choice([False, True], 
                                     p = [1 - self.rate_client_fake, self.rate_client_fake])
        if self.is_malicious:
            
            print(f'malicioso: {self.id}')
            if self.atack == 'zero':
                return model_zeros(self.model, self.device)
            elif self.atack == 'random':
                return random_param(self.model, self.device)
            elif self.atack == 'shuffle':
                return shuffle_model(self.model)
            elif self.atack == 'label':
                return self._train_with_label_flip()
            elif self.atack in ('gaussian', 'gaussian_noise'):
                return gaussian_noise_model(self.model, snr=self.attack_noise_snr)
            elif self.atack in ('sign_flipping', 'sign_flip'):
                return scaled_update_model(self.model, self.latest_global_model, -1.0)
            elif self.atack == 'scaling':
                return scaled_update_model(self.model, self.latest_global_model, self.attack_scale)
            elif self.atack == 'model_replacement':
                return scaled_update_model(self.model, self.latest_global_model, self.attack_scale)
            elif self.atack == 'all':
                numero = random.choice([1, 2, 3, 4])
                if numero == 1:
                    print("ataque zeros")
                    return model_zeros(self.model, self.device)
                elif numero ==2:
                    print("ataque random")
                    return random_param(self.model, self.device)
                elif numero ==3:
                    print("ataque shuffle")
                    return shuffle_model(self.model)
                elif numero==4:
                    return self._train_with_label_flip()

        return self.model
