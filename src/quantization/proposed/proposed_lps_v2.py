import torch
import torch.nn as nn


def compute_rms(weight):
    """
    Compute RMS scaling factor for a weight tensor.
    """
    return torch.sqrt(torch.mean(weight**2) + 1e-8)


def apply_proposed_lps_v2(model, device):
    """
    PTQ++ v2 - Learned Pre-Scaling

    Offline weight conditioning only.

    - Normalizes Conv/Linear weights by RMS
    - Stores RMS value for logging
    - DOES NOT wrap layers
    - DOES NOT modify network architecture
    - Fully compatible with FX quantization
    """

    print("\n========================================")
    print("PTQ++ Stage 1 : Learned Pre-Scaling")
    print("========================================\n")

    model.eval()
    model.to(device)

    total_layers = 0

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            rms = compute_rms(weight)

            module.weight.data = weight / rms

            # Keep for inspection / logging only
            module.register_buffer(
                "ptqpp_rms_scale",
                rms.detach().clone(),
                persistent=True,
            )

            print(f"{name:<40}" f" RMS={rms.item():.6f}")

            total_layers += 1

    print("\n----------------------------------------")
    print(f"LPS applied to {total_layers} layers")
    print("----------------------------------------\n")

    return model
