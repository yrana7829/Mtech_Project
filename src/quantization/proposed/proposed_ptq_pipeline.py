import torch
import torch.nn as nn

from .proposed_lps import apply_proposed_lps, wrap_proposed_lps_layers
from .proposed_twc import apply_proposed_twc
from .proposed_mixed_precision import apply_proposed_mixed_precision


def apply_proposed_ptq_pipeline(model, device):

    print("\nApplying PTQ++ Pipeline...\n")

    # Step 1: LPS
    print("Step 1: Proposed LPS")
    model = apply_proposed_lps(model, device)
    model = wrap_proposed_lps_layers(model)

    # Step 2: Tail Weighted Clipping
    print("Step 2: Tail Weighted Clipping")
    model = apply_proposed_twc(model)

    # Step 3: Mixed Precision
    print("Step 3: Mixed Precision Allocation")
    model = apply_proposed_mixed_precision(model)

    print("\nPTQ++ pipeline completed.\n")

    return model
