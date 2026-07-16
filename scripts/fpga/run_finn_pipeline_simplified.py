import os
import traceback

from qonnx.core.modelwrapper import ModelWrapper

# QONNX
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import (
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes

# FINN
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferThresholdingLayer,
    InferConvInpGen,
    InferPool,
)

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)

# =======================================================
# CHANGE THESE TWO LINES ONLY
# =======================================================

# MODEL = "/home/yrana/project/models/mnv2_eurosat_naive_ptq_int8.onnx"

# RUN_NAME = "naive_ptq"
MODEL = "/home/yrana/project/models/mnv2_eurosat_ptqpp_v3_int8_nofold_sim.onnx"

RUN_NAME = "Proposed PTQ++ No-Fold simplified"

# =======================================================

OUTDIR = f"/home/yrana/project/results/finn/{RUN_NAME}"

os.makedirs(OUTDIR, exist_ok=True)

summary = []


def run_stage(name, transform=None, save_name=None):

    global model

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    try:

        if transform is not None:
            model = model.transform(transform)

        print("PASS")

        summary.append([name, "PASS", ""])

        if save_name is not None:
            outfile = os.path.join(OUTDIR, save_name)
            model.save(outfile)
            print("Saved:", outfile)

    except Exception as e:

        print("FAIL")

        print(e)

        summary.append([name, "FAIL", str(e)])

        traceback.print_exc()

        write_summary()

        exit()


def write_summary():

    txt = os.path.join(OUTDIR, "summary.txt")

    csv = os.path.join(OUTDIR, "summary.csv")

    with open(txt, "w") as f:

        f.write("FINN PIPELINE SUMMARY\n\n")

        for row in summary:
            f.write(f"{row[0]:35s} {row[1]}\n")

            if row[2]:
                f.write(row[2] + "\n")

    with open(csv, "w") as f:

        f.write("Stage,Status,Message\n")

        for row in summary:

            msg = row[2].replace(",", ";")

            f.write(f"{row[0]},{row[1]},{msg}\n")


print("=" * 70)
print("FINN PIPELINE")
print("=" * 70)

print("\nLoading model...")

model = ModelWrapper(MODEL)

summary.append(["Load Model", "PASS", ""])

model.save(os.path.join(OUTDIR, "00_loaded.onnx"))

run_stage(
    "Shape Inference",
    InferShapes(),
    "01_shapes.onnx",
)

run_stage(
    "Fold Constants",
    FoldConstants(),
    "02_fold.onnx",
)

run_stage(
    "Remove Unused Tensors",
    RemoveUnusedTensors(),
    "03_cleanup.onnx",
)

run_stage(
    "Sort Graph",
    SortGraph(),
    "04_sorted.onnx",
)

run_stage(
    "Datatype Inference",
    InferDataTypes(),
    "05_dtype.onnx",
)

run_stage(
    "Unique Parameter Tensors",
    GiveUniqueParameterTensors(),
    "06_unique.onnx",
)

run_stage(
    "Convert QONNX -> FINN",
    ConvertQONNXtoFINN(),
    "07_finn.onnx",
)

run_stage(
    "Infer Threshold Layers",
    InferThresholdingLayer(),
    "08_threshold.onnx",
)

run_stage(
    "Infer Conv Input Generator",
    InferConvInpGen(),
    "09_convinpgen.onnx",
)

run_stage(
    "Infer Pool",
    InferPool(),
    "10_pool.onnx",
)

run_stage(
    "Create Dataflow Partition",
    CreateDataflowPartition(),
    "11_dataflow.onnx",
)

write_summary()

print("\n")
print("=" * 70)
print("PIPELINE FINISHED")
print("=" * 70)

print("\nSummary\n")

for row in summary:
    print(f"{row[0]:35s} {row[1]}")

print("\nResults written to")

print(OUTDIR)
