import torch
import torch.nn as nn


def apply_proposed_lps_v3(model, calibration_loader, device):
    """
    PTQ++ v3

    Activation-aware Learned Pre-Scaling

    Instead of normalizing weights, estimate the RMS
    activation entering each Conv/Linear layer and
    condition the weights accordingly.

    No wrappers.
    No runtime modules.
    Fully FX compatible.
    """

    print("\n========================================")
    print("PTQ++ v3 : Activation-aware LPS")
    print("========================================\n")

    model.eval()
    model.to(device)

    activation_rms = {}
    hooks = []

    # -------------------------------------------------
    # Collect input activation RMS
    # -------------------------------------------------

    def hook_fn(name):

        def fn(module, inputs, outputs):

            x = inputs[0].detach()

            rms = torch.sqrt(torch.mean(x**2) + 1e-8)

            activation_rms[name] = rms.item()

        return fn

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            hooks.append(module.register_forward_hook(hook_fn(name)))

    print("Collecting activation statistics...")

    with torch.no_grad():

        for images, _ in calibration_loader:

            images = images.to(device)

            model(images)

    for h in hooks:
        h.remove()

    # -------------------------------------------------
    # Weight conditioning
    # -------------------------------------------------

    print("\nApplying activation-aware scaling...\n")

    for name, module in model.named_modules():

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        rms = activation_rms.get(name, None)

        if rms is None:
            continue

        scale = 1.0 / (rms + 1e-8)

        module.weight.data.mul_(scale)

        print(f"{name:<40}" f"Activation RMS={rms:.5f}" f" Scale={scale:.5f}")

    print("\n----------------------------------------")
    print("Activation-aware LPS completed")
    print("----------------------------------------\n")

    return model
