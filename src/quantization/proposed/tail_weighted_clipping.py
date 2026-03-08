import torch
import torch.nn as nn


def apply_tail_weighted_clipping(model, percentile=99.0):

    print("\nApplying Tail-Weighted Clipping...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            abs_weight = weight.abs().flatten()

            clip_val = torch.quantile(abs_weight, percentile / 100.0)

            clipped_weight = torch.clamp(weight, -clip_val, clip_val)

            module.weight.data = clipped_weight

            print(f"{name} → clip={clip_val.item():.4f}")

    print("\nTail-Weighted Clipping completed.\n")

    return model
