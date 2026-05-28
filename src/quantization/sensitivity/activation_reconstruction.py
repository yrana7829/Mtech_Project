import torch
import pandas as pd
import numpy as np


class ActivationReconstructionAnalyzer:

    def __init__(self, target_layers, activation_ranges):

        self.target_layers = target_layers

        self.activation_ranges = activation_ranges

        self.fp32_activations = {}

        self.metrics = {}

        self.handles = []

    # ---------------------------------
    # Fake INT8 quantization
    # ---------------------------------
    def quantize_activation(self, x, x_min, x_max, num_bits=8):

        qmin = 0
        qmax = (2**num_bits) - 1

        scale = (x_max - x_min) / (qmax - qmin)

        scale = max(scale, 1e-8)

        zero_point = qmin - round(x_min / scale)

        q_x = torch.round(x / scale + zero_point)

        q_x = torch.clamp(q_x, qmin, qmax)

        dq_x = (q_x - zero_point) * scale

        return dq_x

    # ---------------------------------
    # Register hooks
    # ---------------------------------
    def register_hooks(self, model):

        def hook_fn(name):

            def hook(module, input, output):

                fp32_output = output.detach().cpu()

                x_min = self.activation_ranges[name]["min"]

                x_max = self.activation_ranges[name]["max"]

                quant_output = self.quantize_activation(fp32_output, x_min, x_max)

                error = fp32_output - quant_output

                mse = torch.mean(error**2).item()

                signal_power = torch.mean(fp32_output**2).item()

                noise_power = torch.mean(error**2).item()

                relative_error = (
                    torch.norm(error) / (torch.norm(fp32_output) + 1e-8)
                ).item()

                # SQNR
                sqnr = 10 * np.log10((signal_power + 1e-8) / (noise_power + 1e-8))

                if name not in (self.metrics):

                    self.metrics[name] = {"mse": [], "relative_error": [], "sqnr": []}

                self.metrics[name]["mse"].append(mse)

                self.metrics[name]["relative_error"].append(relative_error)

                self.metrics[name]["sqnr"].append(sqnr)

            return hook

        for name, module in model.named_modules():

            if name in self.target_layers:

                handle = module.register_forward_hook(hook_fn(name))

                self.handles.append(handle)

                print(f"[INFO] Hook attached: " f"{name}")

    # ---------------------------------
    # Remove hooks
    # ---------------------------------
    def remove_hooks(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []

    # ---------------------------------
    # Compute final stats
    # ---------------------------------
    def compute_results(self):

        rows = []

        for layer, stats in self.metrics.items():

            rows.append(
                {
                    "layer": layer,
                    "mse": np.mean(stats["mse"]),
                    "relative_error": np.mean(stats["relative_error"]),
                    "sqnr": np.mean(stats["sqnr"]),
                }
            )

        return pd.DataFrame(rows)
