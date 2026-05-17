import torch
import numpy as np
import pandas as pd
from scipy.stats import kurtosis


class ActivationStatisticsCollector:

    def __init__(self, target_layers):

        self.target_layers = target_layers

        self.activations = {}

        self.handles = []

    def register_hooks(self, model):

        def hook_fn(name):

            def hook(module, input, output):

                activation = output.detach().cpu()

                if name not in self.activations:
                    self.activations[name] = []

                self.activations[name].append(activation.flatten())

            return hook

        for name, module in model.named_modules():

            if name in self.target_layers:

                handle = module.register_forward_hook(hook_fn(name))

                self.handles.append(handle)

                print(f"[INFO] Hook attached: {name}")

    def remove_hooks(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []

    def compute_statistics(self):

        results = []

        for layer_name, tensors in self.activations.items():

            activations = torch.cat(tensors)

            activations_np = activations.numpy()

            # -------------------------
            # Basic statistics
            # -------------------------
            act_min = np.min(activations_np)
            act_max = np.max(activations_np)

            act_mean = np.mean(activations_np)
            act_std = np.std(activations_np)

            act_range = act_max - act_min

            # -------------------------
            # Percentiles
            # -------------------------
            p95 = np.percentile(activations_np, 95)

            p99 = np.percentile(activations_np, 99)

            p999 = np.percentile(activations_np, 99.9)

            # -------------------------
            # Outlier ratio
            # -------------------------
            outlier_ratio = np.sum(activations_np > p99) / len(activations_np)

            # -------------------------
            # Sparsity
            # -------------------------
            sparsity = np.sum(np.abs(activations_np) < 1e-6) / len(activations_np)

            # -------------------------
            # Kurtosis
            # -------------------------
            act_kurtosis = kurtosis(activations_np, fisher=False)

            results.append(
                {
                    "layer": layer_name,
                    "min": act_min,
                    "max": act_max,
                    "range": act_range,
                    "mean": act_mean,
                    "std": act_std,
                    "kurtosis": act_kurtosis,
                    "sparsity": sparsity,
                    "p95": p95,
                    "p99": p99,
                    "p999": p999,
                    "outlier_ratio": outlier_ratio,
                }
            )

        return pd.DataFrame(results)
