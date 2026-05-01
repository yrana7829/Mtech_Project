import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# Quantizer (symmetric)
# ----------------------------------------
def compute_scale(weight, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = weight.abs().max() / qmax + 1e-8
    return scale


# ----------------------------------------
# AdaRound layer
# ----------------------------------------
class AdaRoundLayer(nn.Module):

    def __init__(self, weight, num_bits=8):
        super().__init__()

        self.num_bits = num_bits
        self.qmax = 2 ** (num_bits - 1) - 1

        self.scale = compute_scale(weight, num_bits)

        w_scaled = weight / self.scale

        # floor
        self.w_floor = torch.floor(w_scaled)

        # alpha (learnable)
        self.alpha = nn.Parameter(torch.zeros_like(weight))

    def forward(self):

        # soft rounding
        h = torch.sigmoid(self.alpha)

        w_q = self.w_floor + h

        w_q = torch.clamp(w_q, -self.qmax, self.qmax)

        return w_q * self.scale


# ----------------------------------------
# Optimize one layer
# ----------------------------------------
def optimize_layer(module, input_data, device, num_bits=8, iters=300, lr=1e-2):

    print(f"Optimizing layer: {module}")

    weight = module.weight.data.clone().to(device)

    adaround_layer = AdaRoundLayer(weight, num_bits).to(device)

    optimizer = torch.optim.Adam([adaround_layer.alpha], lr=lr)

    # Freeze original weights
    module.weight.requires_grad = False

    # Capture FP32 output
    with torch.no_grad():
        fp32_out = module(input_data)

    for i in range(iters):

        optimizer.zero_grad()

        # quantized weight
        w_q = adaround_layer()

        # forward with quantized weight
        out_q = F.conv2d(
            input_data,
            w_q,
            bias=module.bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        # reconstruction loss
        loss = F.mse_loss(out_q, fp32_out)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}/{iters}, Loss: {loss.item():.6f}")

    # replace weight
    module.weight.data = adaround_layer().detach()

    return module


# ----------------------------------------
# Apply AdaRound (minimal)
# ----------------------------------------
def apply_adaround(model, calibration_loader, device, num_bits=8):

    model.eval()
    model.to(device)

    print("\nStarting AdaRound (minimal)...\n")

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            print(f"\nProcessing layer: {name}")

            # get one batch
            images, _ = next(iter(calibration_loader))
            images = images.to(device)

            with torch.no_grad():
                model(images)

            # forward to get input to this layer
            def get_input_hook(module, inp, out):
                module.input = inp[0]

            hook = module.register_forward_hook(get_input_hook)

            model(images)

            input_data = module.input.detach()

            hook.remove()

            # optimize this layer
            optimize_layer(module, input_data, device, num_bits)
            torch.cuda.empty_cache()

    print("\nAdaRound complete.\n")

    return model
