import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances


class RepresentationSeparabilityAnalyzer:

    def __init__(self, target_layers, activation_ranges):

        self.target_layers = target_layers
        self.activation_ranges = activation_ranges

        self.fp32_features = {}
        self.quant_features = {}

        self.labels = []

        self.handles = []

        self.quant_mode = False

    # -----------------------------------
    # Fake W32A8 activation quantization
    # -----------------------------------
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

    # -----------------------------------
    # Hooks
    # -----------------------------------
    def register_hooks(self, model):

        def hook_fn(name):

            def hook(module, input, output):

                feat = output.detach().cpu()

                if self.quant_mode:

                    x_min = self.activation_ranges[name]["min"]

                    x_max = self.activation_ranges[name]["max"]

                    feat = self.quantize_activation(feat, x_min, x_max)

                feat = feat.flatten(start_dim=1)

                target_dict = (
                    self.quant_features if self.quant_mode else self.fp32_features
                )

                if name not in target_dict:

                    target_dict[name] = []

                target_dict[name].append(feat.numpy())

            return hook

        for name, module in model.named_modules():

            if name in self.target_layers:

                handle = module.register_forward_hook(hook_fn(name))

                self.handles.append(handle)

                print(f"[INFO] Hook attached: " f"{name}")

    def remove_hooks(self):

        for handle in self.handles:
            handle.remove()

    # -----------------------------------
    # Metrics
    # -----------------------------------
    def compute_metrics(self):

        rows = []

        for layer in self.target_layers:

            fp32 = np.concatenate(self.fp32_features[layer])

            quant = np.concatenate(self.quant_features[layer])

            labels = np.array(self.labels)

            # -------------------
            # Silhouette
            # -------------------
            fp32_sil = silhouette_score(fp32, labels)

            quant_sil = silhouette_score(quant, labels)

            # -------------------
            # Intra/inter
            # -------------------
            fp32_dist = euclidean_distances(fp32)

            quant_dist = euclidean_distances(quant)

            same_class = labels[:, None] == labels[None, :]

            diff_class = ~same_class

            fp32_intra = np.mean(fp32_dist[same_class])

            fp32_inter = np.mean(fp32_dist[diff_class])

            quant_intra = np.mean(quant_dist[same_class])

            quant_inter = np.mean(quant_dist[diff_class])

            rows.append(
                {
                    "layer": layer,
                    "fp32_silhouette": fp32_sil,
                    "quant_silhouette": quant_sil,
                    "silhouette_drop": fp32_sil - quant_sil,
                    "fp32_intra": fp32_intra,
                    "quant_intra": quant_intra,
                    "fp32_inter": fp32_inter,
                    "quant_inter": quant_inter,
                }
            )

        return pd.DataFrame(rows)
