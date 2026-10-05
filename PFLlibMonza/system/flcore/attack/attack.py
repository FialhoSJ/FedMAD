import torch
import torch.nn as nn
import numpy as np
import copy
import sys

def model_zeros(model, device = 'cpu'):
    # Cria uma cópia profunda do modelo para que o modelo original não seja alterado
    copy_model = copy.deepcopy(model)
    for param in copy_model.parameters():
        param.data = torch.zeros_like(param.data, device=device)

    return copy_model

def random_param(model, device = 'cpu', generator=None):
    # Cria uma cópia profunda do modelo para que o modelo original não seja alterado
    copy_model = copy.deepcopy(model)
    for param in copy_model.parameters():
        # gera valores aleatorios para serem utilizados como parametros
        param_random = torch.rand(size=param.shape, generator=generator)
        param.data = param_random.to(device)

    return copy_model

def shuffle_model(model, generator=None):
    # Cria uma cópia profunda do modelo para que o modelo original não seja alterado
    copy_model = copy.deepcopy(model)
    
    # Itera sobre todos os parâmetros do modelo copiado
    for param in copy_model.parameters():
        # Achata o tensor de parâmetros para uma dimensão
        data_flatten = param.data.view(-1)
        
        # Gera uma permutação aleatória dos índices dos elementos
        index_random = torch.randperm(len(data_flatten), generator=generator)
        
        # Aplica a permutação ao tensor achatado
        shuffled_param = data_flatten[index_random.to(data_flatten.device)]
        
        # Redimensiona o tensor embaralhado de volta ao formato original
        param.data = shuffled_param.view(param.data.shape)
    
    return copy_model


def gaussian_noise_model(model, snr=1.0, generator=None):
    """Return a copy with additive Gaussian noise at the requested linear SNR.

    Each tensor's noise power is its signal power divided by ``snr``. An SNR
    of 1.0 therefore adds Gaussian noise with power equal to the parameter
    tensor's power.
    """
    if snr <= 0:
        raise ValueError("snr must be greater than zero")

    copy_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in copy_model.parameters():
            signal_power = torch.mean(param.data.float().square())
            noise_std = torch.sqrt(signal_power / float(snr))
            noise = torch.randn(
                tuple(param.shape), generator=generator, dtype=torch.float32
            ).to(device=param.device, dtype=param.dtype)
            param.add_(noise * noise_std.to(device=param.device, dtype=param.dtype))
    return copy_model


def scaled_update_model(local_model, reference_model, factor):
    """Apply a factor to the local *update* relative to the received model.

    factor=-1 is sign flipping; factor>1 is update scaling/model replacement.
    The function does not alter either input model.
    """
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("update scale must be finite")
    result = copy.deepcopy(local_model)
    reference = reference_model.state_dict()
    with torch.no_grad():
        for name, value in result.state_dict().items():
            if not value.is_floating_point():
                continue
            base = reference[name].to(device=value.device, dtype=value.dtype)
            value.copy_(base + factor * (value - base))
    return result


def shuffle_layer_channels_model(model, generator=None):
    """Permute output channels/units independently in each Conv/Linear layer.

    The layer's bias follows the same permutation. This models the
    Shuffled-Layer/channel-order attack while preserving tensor shapes.
    """
    copy_model = copy.deepcopy(model)
    with torch.no_grad():
        for layer in copy_model.modules():
            if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                continue
            if layer.weight is None or layer.weight.shape[0] < 2:
                continue
            permutation = torch.randperm(layer.weight.shape[0], generator=generator)
            permutation = permutation.to(device=layer.weight.device)
            layer.weight.copy_(layer.weight.index_select(0, permutation))
            if layer.bias is not None:
                layer.bias.copy_(layer.bias.index_select(0, permutation))
    return copy_model

def model_noise(model, SNR, client):
    # Fazendo uma cópia profunda do modelo para preservar o modelo original
    client.train()
    return client.model

    for param in copy_model.parameters():
        with torch.no_grad():
            flatten_param = torch.flatten(param.data)
            num_elements = len(flatten_param)

            #print(param.data[0])
            # Calculando a potência do sinal
            power_signal = torch.mean(param.data ** 2)

            # Calculando a potência do ruído com base na SNR desejada
            power_noise =  power_signal / SNR
            
            #print(f"Potência do ruído: {power_noise.item()}")

            # Gerando o ruído com a distribuição normal com desvio padrão adequado
            noise_matrix = np.random.normal(0.0, np.sqrt(power_noise), size = param.data.shape)

            # Adicionando o ruído ao parâmetro original
            param.add_(torch.tensor(noise_matrix))

            # Verificação da potência real do ruído (opcional)
            #actual_power_noise = torch.sum(noise_matrix ** 2) / num_elements
            #actual_power_signal = torch.sum(param.data ** 2) / num_elements
            #print(f"Potência real do ruído: {actual_power_noise.item()}")
            #print(f"Potência final do sinal: {actual_power_signal.item()}")

            # Calculando a SNR real (opcional)
            snr_real = (power_signal / noise_matrix)
            #print(f"SNR real calculado: {snr_real.item()}")


    return copy_model

#if __name__ == "__main__":
    #model = nn.Linear(3, 2)

    
    #zero_model = model_zeros(model)
    #random_model = random_param(model)
    #shuffle_model_ = shuffle_model(model)
    #noise_model_ = model_noise(model, 1)

    '''print("Current Model: ")
    for param in model.parameters():
        print(param)

    print("\nShuffle Model: ")
    for param in shuffle_model_.parameters():
        print(param)

    print("\nRandom Model: ")
    for param in random_model.parameters():
        print(param)

    print("\nNoise Model: ")
    for param in noise_model_.parameters():
        print(param)

    print("\nZero Model: ")
    for param in zero_model.parameters():
        print(param)'''
