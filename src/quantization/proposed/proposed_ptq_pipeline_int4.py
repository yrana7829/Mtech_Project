import torch
import torch.nn as nn

from .proposed_lps import apply_proposed_lps, wrap_proposed_lps_layers
from .proposed_twc import apply_proposed_twc
from .proposed_mixed_precision_int4 import apply_proposed_mixed_precision_int4


def apply_proposed_ptq_pipeline_int4(model, device):

    print("\nApplying PTQ++ INT4 Pipeline...\n")

    print("Step 1: Proposed LPS")
    model = apply_proposed_lps(model, device)
    model = wrap_proposed_lps_layers(model)

    print("Step 3: Tail Weighted Clipping")
    model = apply_proposed_twc(model)

    print("Step 2: Mixed Precision INT4")
    model = apply_proposed_mixed_precision_int4(model)

    print("\nPTQ++ INT4 pipeline completed.\n")

    return model
