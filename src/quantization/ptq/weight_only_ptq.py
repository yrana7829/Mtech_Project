import copy
import torch
import torch.nn as nn


def quantize_tensor_symmetric(tensor, num_bits=8):

    qmax = (2 ** (num_bits - 1)) - 1

    max_val = tensor.abs().max()

    if max_val < 1e-8:
        return tensor

    scale = max_val / qmax

    q_tensor = torch.round(tensor / scale)

    q_tensor = torch.clamp(q_tensor, -qmax, qmax)

    dq_tensor = q_tensor * scale

    return dq_tensor


def apply_weight_only_quantization(model, num_bits=8):

    quantized_model = copy.deepcopy(model)

    quantized_model.eval()

    for name, module in quantized_model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            print(f"[INFO] Quantizing weights: {name}")

            with torch.no_grad():

                quantized_weight = quantize_tensor_symmetric(
                    module.weight.data, num_bits=num_bits
                )

                module.weight.data.copy_(quantized_weight)

    return quantized_model
