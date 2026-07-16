from qonnx.core.modelwrapper import ModelWrapper
from collections import Counter

# -------------------------------------------------------
# CHANGE MODEL PATH IF REQUIRED
# -------------------------------------------------------

MODEL = "/home/yrana/project/results/finn/Proposed PTQ++ No-Fold/07_finn.onnx"

# -------------------------------------------------------

model = ModelWrapper(MODEL)

producer = {}
consumer = {}

for node in model.graph.node:

    for out in node.output:
        producer[out] = node

    for inp in node.input:
        consumer.setdefault(inp, []).append(node)

patterns = Counter()

for node in model.graph.node:
    if node.op_type != "Cast":
        continue

    prev_ops = []
    next_ops = []

    for inp in node.input:
        if inp in producer:
            prev_ops.append(producer[inp].op_type)
        else:
            prev_ops.append("GraphInput")

    for out in node.output:
        if out in consumer:
            for c in consumer[out]:
                next_ops.append(c.op_type)
        else:
            next_ops.append("GraphOutput")

    for p in prev_ops:
        for n in next_ops:
            patterns[(p, n)] += 1

print("\nPATTERN SUMMARY")
for (p, n), c in sorted(patterns.items()):
    print(f"{p:25s} -> {n:25s} : {c}")

for k, v in sorted(patterns.items()):
    print(f"{k[0]:25s} -> {k[1]:25s} : {v}")
