import copy
import torch
import torch.nn as nn


class StaticActivationQuantizer(nn.Module):

    def __init__(self, min_val, max_val, num_bits=8):

        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

        self.num_bits = num_bits

        self.qmin = 0
        self.qmax = (2**num_bits) - 1

    def forward(self, x):

        min_val = self.min_val
        max_val = self.max_val

        if (max_val - min_val) < 1e-8:
            return x

        scale = (max_val - min_val) / (self.qmax - self.qmin)

        zero_point = self.qmin - min_val / scale

        q_x = torch.round(x / scale + zero_point)

        q_x = torch.clamp(q_x, self.qmin, self.qmax)

        dq_x = (q_x - zero_point) * scale

        return dq_x


def collect_activation_ranges(model, calibration_loader):

    activation_stats = {}

    hooks = []

    def register_hook(name):

        def hook(module, input, output):

            x_min = output.min().item()
            x_max = output.max().item()

            if name not in activation_stats:

                activation_stats[name] = {"min": x_min, "max": x_max}

            else:

                activation_stats[name]["min"] = min(
                    activation_stats[name]["min"], x_min
                )

                activation_stats[name]["max"] = max(
                    activation_stats[name]["max"], x_max
                )

        return hook

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6)):

            hooks.append(module.register_forward_hook(register_hook(name)))

    model.eval()

    with torch.no_grad():

        for images, _ in calibration_loader:

            model(images.cpu())

    for hook in hooks:
        hook.remove()

    return activation_stats


class StaticActivationSensitivityAnalyzer:

    def __init__(self, model, activation_stats, target_module_names, num_bits=8):

        self.model = model

        self.activation_stats = activation_stats

        self.target_module_names = target_module_names

        self.num_bits = num_bits

        self.handles = []

    def register_hooks(self):

        for name, module in self.model.named_modules():

            if name in self.target_module_names:

                stats = self.activation_stats[name]

                quantizer = StaticActivationQuantizer(
                    stats["min"], stats["max"], self.num_bits
                )

                def hook_fn(module, input, output, quantizer=quantizer):

                    return quantizer(output)

                handle = module.register_forward_hook(hook_fn)

                self.handles.append(handle)

                print(f"[INFO] Hook attached to: {name}")

    def remove_hooks(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []
