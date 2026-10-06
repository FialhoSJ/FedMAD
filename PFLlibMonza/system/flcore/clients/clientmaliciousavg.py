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
        self.source_label = getattr(args, "attack_source_label", None)
        self.target_label = getattr(args, "attack_target_label", None)
        self.last_attack = None
        self.v2 = getattr(args, "mad_version", "v1") == "v2"
        self._label_attack_decision = None
        if (self.source_label is None) != (self.target_label is None):
            raise ValueError("targeted label poisoning requires both source and target labels")
        if self.source_label is not None and (self.source_label == self.target_label or not 0 <= self.source_label < self.num_classes or not 0 <= self.target_label < self.num_classes):
            raise ValueError("source and target must be distinct valid class indices")

    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def set_parameters(self, model):
        self.latest_global_model = model
        return super().set_parameters(model)

    def _attack_seed(self, round_number):
        return (self.mad_seed * 1000003 + (int(round_number) + 1) * 100003 + self.id * 1009 + 7919) % (2**63 - 1)

    def train(self):
        """V2 label poisoning uses the same number of local steps as benign training."""
        round_number = self._mad_round
        if self.v2 and self.atack in ('label', 'label_flipping') and round_number > self.round_init_atk:
            active = (random.Random(self._attack_seed(round_number)).random() < self.rate_client_fake
                      if self.mad_deterministic else bool(np.random.choice([False, True], p=[1 - self.rate_client_fake, self.rate_client_fake])))
            self._label_attack_decision = (round_number, active)
            if active:
                start_time = time.time()
                if self.source_label is None:
                    self._train_with_label_flip()
                else:
                    self._train_with_targeted_label_flip(self.source_label, self.target_label)
                if self.learning_rate_decay:
                    self.learning_rate_scheduler.step()
                self.train_time_cost['num_rounds'] += 1
                self.train_time_cost['total_cost'] += time.time() - start_time
                return
        super().train()

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
        self.is_malicious = False
        self.last_attack = None
        if round <= self.round_init_atk:
            return self.model
        generator = None
        if self.mad_deterministic:
            draw_seed = self._attack_seed(round)
            rng = random.Random(draw_seed)
            self.is_malicious = rng.random() < self.rate_client_fake
            generator = torch.Generator().manual_seed(draw_seed)
        else:
            rng = random
            self.is_malicious = np.random.choice([False, True],
                                     p = [1 - self.rate_client_fake, self.rate_client_fake])
        label_decision = getattr(self, '_label_attack_decision', None)
        if label_decision is not None and label_decision[0] == round:
            self.is_malicious = label_decision[1]
        if self.is_malicious:
            self.last_attack = self.atack
            
            print(f'malicioso: {self.id}')
            if self.atack == 'zero':
                return model_zeros(self.model, self.device)
            elif self.atack == 'random':
                return random_param(self.model, self.device, generator=generator)
            elif self.atack == 'shuffle':
                return shuffle_model(self.model, generator=generator)
            elif self.atack in ('label', 'label_flipping'):
                if label_decision is not None and label_decision[0] == round:
                    return self.model
                if self.source_label is not None:
                    return self._train_with_targeted_label_flip(self.source_label, self.target_label)
                return self._train_with_label_flip()
            elif self.atack in ('gaussian', 'gaussian_noise'):
                return gaussian_noise_model(self.model, snr=self.attack_noise_snr, generator=generator)
            elif self.atack in ('sign_flipping', 'sign_flip'):
                return scaled_update_model(self.model, self.latest_global_model, -1.0)
            elif self.atack == 'scaling':
                return scaled_update_model(self.model, self.latest_global_model, self.attack_scale)
            elif self.atack == 'model_replacement':
                return scaled_update_model(self.model, self.latest_global_model, self.attack_scale)
            elif self.atack == 'all':
                numero = rng.choice([1, 2, 3, 4])
                self.last_attack = {1: 'zero', 2: 'random', 3: 'shuffle', 4: 'label'}[numero]
                if numero == 1:
                    print("ataque zeros")
                    return model_zeros(self.model, self.device)
                elif numero ==2:
                    print("ataque random")
                    return random_param(self.model, self.device, generator=generator)
                elif numero ==3:
                    print("ataque shuffle")
                    return shuffle_model(self.model, generator=generator)
                elif numero==4:
                    return self._train_with_label_flip()
            else:
                raise ValueError(f"unsupported attack: {self.atack}")

        return self.model
