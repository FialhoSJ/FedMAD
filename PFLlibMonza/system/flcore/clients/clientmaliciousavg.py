import torch
import torch.nn as nn
import numpy as np
import time
import random
from flcore.clients.clientbase import Client
from flcore.clients.clientavg import clientAVG

#from utils.privacy import *

from flcore.attack.attack import *


class ClientMaliciousAVG(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.rate_client_fake = args.rate_client_fake
        self.atack = args.atack

        self.list_global_model = []
        #self.delay_atk = args.delay_atk
        self.round_init_atk = args.round_init_atk
        self.model = copy.deepcopy(args.model)

    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def set_parameters(self, model):
        self.list_global_model.append(model)
        return super().set_parameters(model)
    
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
                is_malicious = True
                trainloader = self.load_train_data(None, is_malicious)
        # self.model.to(self.device)
                self.model.train()
                start_time = time.time()

                max_local_epochs = self.local_epochs
                if self.train_slow:
                    max_local_epochs = np.random.randint(1, max_local_epochs // 2)

                for epoch in range(max_local_epochs):
                    for i, (x, y) in enumerate(trainloader):
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)
                        if self.train_slow:
                            time.sleep(0.1 * np.abs(np.random.rand()))
                        output = self.model(x)
                        loss = self.loss(output, y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                return self.model
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
                    is_malicious = True
                    trainloader = self.load_train_data(None, is_malicious)
                    # self.model.to(self.device)
                    self.model.train()
                    max_local_epochs = self.local_epochs
                    if self.train_slow:
                        max_local_epochs = np.random.randint(1, max_local_epochs // 2)

                    for epoch in range(max_local_epochs):
                        for i, (x, y) in enumerate(trainloader):
                            if type(x) == type([]):
                                x[0] = x[0].to(self.device)
                            else:
                                x = x.to(self.device)
                            y = y.to(self.device)
                            if self.train_slow:
                                time.sleep(0.1 * np.abs(np.random.rand()))
                            output = self.model(x)
                            loss = self.loss(output, y)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                    return self.model

        return self.model