import torch
import torch.nn as nn


def compute_clip_threshold(weight, percentile=99.0):

    abs_weight = weight.abs().flatten()

    threshold = torch.quantile(abs_weight, percentile / 100.0)

    return threshold


def apply_tail_weighted_clipping(model, percentile=99.0):

    print("\nApplying Tail-Weighted Clipping...\n")

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            weight = module.weight.data

            clip_val = compute_clip_threshold(weight, percentile)

            clipped = torch.clamp(weight, -clip_val, clip_val)

            module.weight.data = clipped

            print(f"{name}  clip={clip_val:.4f}")

    return model
