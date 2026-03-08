import torch
import torch.nn as nn


def apply_tail_weighted_clipping(model, percentile=99.5):

    print("\nApplying Tail-Weighted Clipping...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            # compute clipping threshold
            threshold = torch.quantile(weight.abs(), percentile / 100.0)

            clipped_weight = torch.clamp(weight, -threshold, threshold)

            module.weight.data = clipped_weight

            print(f"{name} → clip_threshold={threshold.item():.4f}")

    print("\nTail-Weighted Clipping completed.\n")

    return model
