import torch
import torch.nn as nn


def get_layer_percentile(layer_name):
    """
    Layer-wise clipping policy derived from the
    sensitivity analysis.

    Early layers:
        Preserve almost everything.

    Middle layers:
        Mild clipping.

    Late layers:
        Slightly stronger clipping because feature
        drift increases toward the end of MobileNetV2.
    """

    if layer_name.startswith("features."):

        idx = int(layer_name.split(".")[1])

        # Early feature extractor
        if idx <= 5:
            return 99.95

        # Middle bottlenecks
        elif idx <= 12:
            return 99.95

        # Late bottlenecks
        else:
            return 99.90

    # classifier
    return 99.90


def apply_proposed_twc_v3(model):

    print("\n========================================")
    print("PTQ++ v3 : Layer-wise Tail Weighted Clipping")
    print("========================================\n")

    total = 0

    for name, module in model.named_modules():

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        percentile = get_layer_percentile(name)

        weight = module.weight.data

        threshold = torch.quantile(
            weight.abs(),
            percentile / 100.0,
        )

        before = weight.abs().max().item()

        module.weight.data.clamp_(
            -threshold,
            threshold,
        )

        after = module.weight.data.abs().max().item()

        print(
            f"{name:<40}"
            f"P={percentile:<6}"
            f" MaxBefore={before:.5f}"
            f" MaxAfter={after:.5f}"
        )

        total += 1

    print("\n----------------------------------------")
    print(f"TWC applied to {total} layers")
    print("----------------------------------------\n")

    return model
