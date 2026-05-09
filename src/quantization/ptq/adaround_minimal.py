import torch
import torch.nn as nn
import torch.nn.functional as F

# additional imports for FX implimentation
from torch.ao.quantization import MinMaxObserver, get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import MinMaxObserver


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

            # forward to get input to this layer
            def get_input_hook(module, inp, out):
                module.input = inp[0]

            hook = module.register_forward_hook(get_input_hook)

            with torch.no_grad():
                model(images)
            input_data = module.input.detach().clone()

            hook.remove()

            # optimize this layer
            optimize_layer(module, input_data, device, num_bits)
            torch.cuda.empty_cache()

    print("\nAdaRound complete.\n")

    return model


# Modified FX funtion
def fx_quantize_model(model, calibration_loader, device):

    model.eval()
    model.to(device)

    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8),
    )

    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # example input
    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("\nPreparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

    print("Running calibration...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)

    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model
