import torch
import torch.nn as nn

PERCENTILE = 99.75


def apply_proposed_twc_v2(model):
    """
    PTQ++ v2 - Tail Weighted Clipping

    Offline weight conditioning.

    Performs percentile clipping on Conv/Linear
    weights before PTQ calibration.
    """

    print("\n========================================")
    print("PTQ++ Stage 2 : Tail Weighted Clipping")
    print("========================================\n")

    total_layers = 0

    for name, module in model.named_modules():

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        weight = module.weight.data

        num_params = weight.numel()

        # Same adaptive policy as original implementation
        # if num_params > 50000:
        #     percentile = 99.8
        # else:
        #     percentile = 99.9
        percentile = PERCENTILE

        threshold = torch.quantile(
            weight.abs(),
            percentile / 100.0,
        )

        before_max = weight.abs().max().item()

        module.weight.data.clamp_(
            -threshold,
            threshold,
        )

        after_max = module.weight.data.abs().max().item()

        print(
            f"{name:<40}"
            f" Params={num_params:<8}"
            f" Percentile={percentile:<5}"
            f" Threshold={threshold.item():.6f}"
        )

        print(
            f"{'':40}" f" Max(before)={before_max:.6f}" f" Max(after)={after_max:.6f}"
        )

        total_layers += 1

    print("\n----------------------------------------")
    print(f"TWC applied to {total_layers} layers")
    print("----------------------------------------\n")

    return model
