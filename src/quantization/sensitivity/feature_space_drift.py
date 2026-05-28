import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


class FeatureSpaceDriftAnalyzer:

    def __init__(self, target_layers, activation_ranges):

        self.target_layers = target_layers

        self.activation_ranges = activation_ranges

        self.fp32_features = {}
        self.quant_features = {}

        self.handles = []

        self.results = {}

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
    # Hook registration
    # ---------------------------------
    def register_hooks(self, model):

        def hook_fn(name):

            def hook(module, input, output):

                fp32_output = output.detach().cpu()

                x_min = self.activation_ranges[name]["min"]

                x_max = self.activation_ranges[name]["max"]

                quant_output = self.quantize_activation(fp32_output, x_min, x_max)

                fp32_flat = fp32_output.flatten(start_dim=1)

                quant_flat = quant_output.flatten(start_dim=1)

                # -----------------------
                # Relative error
                # -----------------------
                relative_error = torch.norm(fp32_flat - quant_flat, dim=1) / (
                    torch.norm(fp32_flat, dim=1) + 1e-8
                )

                # -----------------------
                # Cosine similarity
                # -----------------------
                cosine_similarity = F.cosine_similarity(fp32_flat, quant_flat, dim=1)

                if name not in (self.results):

                    self.results[name] = {"relative_error": [], "cosine_similarity": []}

                self.results[name]["relative_error"].extend(
                    relative_error.numpy().tolist()
                )

                self.results[name]["cosine_similarity"].extend(
                    cosine_similarity.numpy().tolist()
                )

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
    # Results
    # ---------------------------------
    def compute_results(self):

        rows = []

        for layer, stats in self.results.items():

            rows.append(
                {
                    "layer": layer,
                    "relative_error": np.mean(stats["relative_error"]),
                    "cosine_similarity": np.mean(stats["cosine_similarity"]),
                }
            )

        return pd.DataFrame(rows)
