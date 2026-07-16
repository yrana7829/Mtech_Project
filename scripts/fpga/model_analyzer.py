#!/usr/bin/env python3
"""
===========================================================
ONNX / QONNX Model Analyzer
Author : Yatendra Rana (IIT Madras M.Tech Thesis)
Purpose: Generic analyzer for ONNX models prior to FINN flow
===========================================================
"""

import os
import argparse
from collections import Counter

import onnx
from onnx import numpy_helper
from qonnx.core.modelwrapper import ModelWrapper

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------


def separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def tensor_shape(tensor):
    try:
        return [d.dim_value for d in tensor.type.tensor_type.shape.dim]
    except Exception:
        return "Unknown"


def human_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)

    for u in units:
        if size < 1024:
            return f"{size:.2f} {u}"
        size /= 1024

    return f"{size:.2f} TB"


# ----------------------------------------------------------
# Analyzer
# ----------------------------------------------------------


def analyze_model(model_path):

    separator("GENERAL INFORMATION")

    model = ModelWrapper(model_path)
    proto = onnx.load(model_path)

    print(f"Model Path      : {model_path}")
    print(f"Model Name      : {os.path.basename(model_path)}")
    print(f"Model Size      : {human_size(os.path.getsize(model_path))}")

    print(f"IR Version      : {proto.ir_version}")
    print(f"Producer        : {proto.producer_name}")
    print(f"Producer Version: {proto.producer_version}")

    if len(proto.opset_import):
        print(f"Opset Version   : {proto.opset_import[0].version}")

    # ------------------------------------------------------

    separator("GRAPH SUMMARY")

    print(f"Nodes           : {len(model.graph.node)}")
    print(f"Inputs          : {len(model.graph.input)}")
    print(f"Outputs         : {len(model.graph.output)}")
    print(f"Initializers    : {len(model.graph.initializer)}")

    # ------------------------------------------------------

    separator("INPUT TENSORS")

    for inp in model.graph.input:
        print(f"Name            : {inp.name}")
        print(f"Shape           : {tensor_shape(inp)}")
        print()

    separator("OUTPUT TENSORS")

    for out in model.graph.output:
        print(f"Name            : {out.name}")
        print(f"Shape           : {tensor_shape(out)}")
        print()

    # ------------------------------------------------------

    separator("OPERATOR STATISTICS")

    op_counts = Counter(node.op_type for node in model.graph.node)

    for op in sorted(op_counts):
        print(f"{op:<25}{op_counts[op]}")

    # ------------------------------------------------------

    separator("CONVOLUTION ANALYSIS")

    conv_count = 0
    depthwise = 0
    pointwise = 0

    for node in model.graph.node:

        if node.op_type != "Conv":
            continue

        conv_count += 1

        attrs = {}

        for a in node.attribute:
            attrs[a.name] = onnx.helper.get_attribute_value(a)

        kernel = attrs.get("kernel_shape", [])
        stride = attrs.get("strides", [])
        pads = attrs.get("pads", [])
        groups = attrs.get("group", 1)

        if kernel == [1, 1]:
            pointwise += 1

        if groups > 1:
            depthwise += 1

        print("-" * 70)
        print(f"Node     : {node.name}")
        print(f"Kernel   : {kernel}")
        print(f"Stride   : {stride}")
        print(f"Padding  : {pads}")
        print(f"Groups   : {groups}")

    print("\nSummary")
    print(f"Total Conv Layers : {conv_count}")
    print(f"Pointwise Conv    : {pointwise}")
    print(f"Grouped/Depthwise : {depthwise}")

    # ------------------------------------------------------

    separator("WEIGHT STATISTICS")

    total_params = 0
    total_memory = 0

    largest_tensor = ("", 0)

    for init in model.graph.initializer:

        arr = numpy_helper.to_array(init)

        params = arr.size
        mem = arr.nbytes

        total_params += params
        total_memory += mem

        if mem > largest_tensor[1]:
            largest_tensor = (init.name, mem)

    print(f"Total Parameters : {total_params:,}")
    print(f"Weight Memory    : {human_size(total_memory)}")
    print(f"Largest Tensor   : {largest_tensor[0]}")
    print(f"Largest Size     : {human_size(largest_tensor[1])}")

    # ------------------------------------------------------

    separator("QUANTIZATION ANALYSIS")

    quant_ops = [
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "MultiThreshold",
    ]

    found = False

    for op in quant_ops:
        if op in op_counts:
            print(f"{op:<25}{op_counts[op]}")
            found = True

    if not found:
        print("No explicit quantization operators found.")
        print("Likely FP32 graph.")

    # ------------------------------------------------------

    separator("FINN COMPATIBILITY (PRELIMINARY)")

    supported = {
        "Conv",
        "Add",
        "Flatten",
        "GlobalAveragePool",
        "Gemm",
        "Constant",
    }

    for op in sorted(op_counts):

        if op in supported:
            print(f"[ OK ] {op}")

        elif op == "Clip":
            print(f"[WARN] Clip (ReLU6) -> Verify transformation")

        else:
            print(f"[ ??? ] {op}")

    # ------------------------------------------------------

    separator("ANALYSIS COMPLETE")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Path to ONNX model")

    args = parser.parse_args()

    analyze_model(args.model)
