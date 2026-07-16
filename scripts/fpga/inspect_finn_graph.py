from qonnx.core.modelwrapper import ModelWrapper
from collections import Counter

MODEL = "/home/yrana/project/results/finn/Proposed PTQ++ No-Fold/11_dataflow.onnx"
# Change this path if needed

model = ModelWrapper(MODEL)

print("=" * 80)
print("FINN GRAPH INSPECTION")
print("=" * 80)

print("\nModel Inputs:")
for i in model.graph.input:
    print(" ", i.name)

print("\nModel Outputs:")
for o in model.graph.output:
    print(" ", o.name)

print("\n")

counter = Counter()

partition_nodes = []

for idx, node in enumerate(model.graph.node):

    counter[node.op_type] += 1

    if node.op_type == "StreamingDataflowPartition":
        partition_nodes.append((idx, node))

print("=" * 80)
print("Operator Summary")
print("=" * 80)

for k in sorted(counter.keys()):
    print(f"{k:35s} {counter[k]}")

print()

print("=" * 80)
print("StreamingDataflowPartition Nodes")
print("=" * 80)

print("Count :", len(partition_nodes))
print()

for idx, node in partition_nodes:

    print("Node Index :", idx)
    print("Name       :", node.name)

    print("\nInputs")
    for x in node.input:
        print("   ", x)

    print("\nOutputs")
    for y in node.output:
        print("   ", y)

    print("-" * 70)
