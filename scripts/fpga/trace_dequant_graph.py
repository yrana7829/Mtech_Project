from qonnx.core.modelwrapper import ModelWrapper

# ----------------------------------------------------------
# CHANGE THIS TO THE MODEL YOU WANT TO ANALYZE
# ----------------------------------------------------------

MODEL = "/home/yrana/project/results/finn/Proposed PTQ++ No-Fold/07_finn.onnx"

# ----------------------------------------------------------

model = ModelWrapper(MODEL)

# ----------------------------------------------------------

producer = {}
consumer = {}

for node in model.graph.node:

    for out in node.output:
        producer[out] = node

    for inp in node.input:
        consumer.setdefault(inp, []).append(node)

# ----------------------------------------------------------

count = 0

print("=" * 90)
print("DEQUANTIZE GRAPH TRACE")
print("=" * 90)

for node in model.graph.node:

    if node.op_type != "DequantizeLinear":
        continue

    count += 1

    print("\n")
    print("=" * 90)
    print(f"Dequantize #{count}")

    print("\nCurrent")
    print("   ", node.op_type)

    print("\nPrevious")

    for inp in node.input:

        if inp in producer:

            p = producer[inp]

            print(f"   {p.op_type:25s} {p.name}")

        else:

            print("   Graph Input")

    print("\nNext")

    for out in node.output:

        if out in consumer:

            for c in consumer[out]:

                print(f"   {c.op_type:25s} {c.name}")

        else:

            print("   Graph Output")

print("\n")
print("=" * 90)
print("Total DequantizeLinear Nodes :", count)

from collections import Counter

patterns = Counter()

for node in model.graph.node:

    if node.op_type != "DequantizeLinear":
        continue

    prev = "Input"
    nxt = "Output"

    for inp in node.input:
        if inp in producer:
            prev = producer[inp].op_type

    for out in node.output:
        if out in consumer:
            nxt = consumer[out][0].op_type

    patterns[(prev, nxt)] += 1

print("\n")
print("=" * 90)
print("PATTERN SUMMARY")
print("=" * 90)

for k, v in patterns.items():
    print(f"{k[0]:20s} -> {k[1]:20s} : {v}")
