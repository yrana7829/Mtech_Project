import torch
import torch.nn as nn


def apply_proposed_twc(model):

    print("\nApplying Proposed Tail-Weighted Clipping...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data
            num_params = weight.numel()

            # adaptive percentile
            if num_params > 50000:
                percentile = 99.7
            else:
                percentile = 99.8

            threshold = torch.quantile(weight.abs(), percentile / 100.0)

            module.weight.data = torch.clamp(weight, -threshold, threshold)

            print(f"{name} → percentile={percentile}")

    print("\nProposed TWC completed.\n")

    return model
