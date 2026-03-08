import torch
import torch.nn as nn
import time


def count_parameters(model):

    return sum(p.numel() for p in model.parameters())


def compute_model_size(model, default_bits=32):

    total_params = 0
    total_bits = 0

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data
            params = weight.numel()

            if hasattr(module, "num_bits"):
                bits = module.num_bits
            else:
                bits = default_bits

            total_params += params
            total_bits += params * bits

    size_mb = total_bits / 8 / 1024 / 1024

    return size_mb


def compute_average_bitwidth(model):

    total_params = 0
    total_bits = 0

    for module in model.modules():

        if hasattr(module, "num_bits"):

            params = module.weight.numel()
            bits = module.num_bits

            total_params += params
            total_bits += params * bits

    if total_params == 0:
        return 32

    return total_bits / total_params


def measure_latency(model, loader, device, runs=30):

    model.eval()

    data_iter = iter(loader)
    images, _ = next(data_iter)
    images = images.to(device)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(images)

    start = time.time()

    for _ in range(runs):
        with torch.no_grad():
            model(images)

    end = time.time()

    return (end - start) / runs
