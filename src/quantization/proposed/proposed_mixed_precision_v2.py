import torch
import torch.nn as nn


def assign_bitwidth(module):
    """
    Simple mixed-precision policy.

    Larger layers:
        -> INT8

    Smaller layers:
        -> INT6 (recommendation only)
    """

    num_params = module.weight.numel()

    if num_params > 50000:
        return 8

    return 6


def apply_proposed_mixed_precision_v2(model):
    """
    PTQ++ v2

    Mixed Precision Allocation

    NOTE:
    This stage DOES NOT quantize the weights.

    It only assigns the recommended precision for each layer.
    """

    print("\n========================================")
    print("PTQ++ Stage 3 : Mixed Precision Allocation")
    print("========================================\n")

    total_layers = 0
    int8_layers = 0
    int6_layers = 0

    for name, module in model.named_modules():

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        bits = assign_bitwidth(module)

        # Store recommendation
        module.ptqpp_bits = bits

        variance = module.weight.data.var().item()

        print(
            f"{name:<40}"
            f" Params={module.weight.numel():<8}"
            f" Variance={variance:.6f}"
            f" -> {bits}-bit"
        )

        if bits == 8:
            int8_layers += 1
        else:
            int6_layers += 1

        total_layers += 1

    print("\n----------------------------------------")
    print(f"Layers analysed : {total_layers}")
    print(f"INT8 layers     : {int8_layers}")
    print(f"INT6 layers     : {int6_layers}")
    print("----------------------------------------\n")

    return model
