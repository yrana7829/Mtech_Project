import torch
from torch.ao.quantization.observer import HistogramObserver


class PercentileObserver(HistogramObserver):
    def __init__(self, percentile=99.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percentile = percentile

    def calculate_qparams(self):
        # Get histogram data
        hist = self.histogram
        bin_edges = self.bin_edges

        # Compute cumulative distribution
        cdf = torch.cumsum(hist, dim=0)
        cdf = cdf / cdf[-1]

        # Find percentile index
        threshold_idx = torch.searchsorted(cdf, self.percentile / 100.0)

        threshold_idx = min(threshold_idx.item(), len(bin_edges) - 2)

        # Get corresponding value
        max_val = bin_edges[threshold_idx + 1]
        min_val = -max_val  # symmetric

        return self._calculate_qparams(min_val, max_val)
