def compute_outlier_ratios(stats):

    ratios = {}

    for layer, values in stats.items():

        max_val = abs(values["max"])
        p99 = abs(values["p99"])

        if p99 == 0:
            ratio = 0
        else:
            ratio = max_val / p99

        ratios[layer] = ratio

    return ratios


def select_layers_for_clipping(ratios, threshold=8):

    selected = []

    for layer, r in ratios.items():

        if r > threshold:
            selected.append(layer)

    return selected
