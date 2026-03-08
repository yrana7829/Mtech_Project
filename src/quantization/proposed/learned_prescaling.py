import torch
import torch.nn as nn


def compute_optimal_scale(weight, num_bits=8):

    qmax = 2 ** (num_bits - 1) - 1

    best_alpha = 1.0
    best_error = float("inf")

    # ensure computation happens on same device
    device = weight.device

    for alpha in torch.linspace(0.5, 2.0, steps=20, device=device):

        scaled = weight * alpha

        scale = scaled.abs().max() / qmax + 1e-8

        q = torch.round(scaled / scale)
        q = torch.clamp(q, -qmax, qmax)

        dequant = q * scale

        recovered = dequant / alpha

        error = torch.mean((weight - recovered) ** 2)

        if error < best_error:
            best_error = error
            best_alpha = alpha.item()

    return best_alpha


def apply_learned_prescaling(model, device):

    model.eval()
    model.to(device)

    print("\nApplying Learned Pre-Scaling...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            alpha = compute_optimal_scale(weight)

            module.weight.data = weight * alpha

            print(f"{name}  → scale={alpha:.3f}")

    print("\nLearned Pre-Scaling completed.\n")

    return model
