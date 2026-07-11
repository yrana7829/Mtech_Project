import torch
import torch.nn as nn


def get_layer_precision(layer_name):
    """
    Sensitivity-driven precision allocation.

    Based on the feature drift analysis:

    Early layers:
        Least sensitive

    Middle layers:
        Moderately sensitive

    Late bottlenecks:
        Most sensitive
    """

    if layer_name.startswith("features."):

        idx = int(layer_name.split(".")[1])

        # Early feature extractor
        if idx <= 5:
            return 6

        # Middle bottlenecks
        elif idx <= 12:
            return 7

        # Late bottlenecks
        else:
            return 8

    # Classifier
    return 8


def fake_quantize_weight(weight, bits):
    """
    Simulate mixed precision by quantizing
    weights to different bit-widths.

    This is still FX-compatible because the
    resulting weights remain FP32.
    """

    qmax = (2 ** (bits - 1)) - 1

    scale = weight.abs().max() / qmax

    if scale < 1e-8:
        return weight

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_proposed_mpa_v3(model):

    print("\n========================================")
    print("PTQ++ v3 : Sensitivity-driven Mixed Precision")
    print("========================================\n")

    total = 0

    for name, module in model.named_modules():

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        bits = get_layer_precision(name)

        module.weight.data = fake_quantize_weight(
            module.weight.data,
            bits,
        )

        print(f"{name:<40}" f"Assigned Precision = INT{bits}")

        total += 1

    print("\n----------------------------------------")
    print(f"Processed {total} layers")
    print("----------------------------------------\n")

    return model
