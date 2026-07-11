from torch.ao.quantization.observer import ObserverBase


def summarize_observers(model):

    total = 0
    initialized = 0
    uninitialized = 0

    for module in model.modules():

        if isinstance(module, ObserverBase):

            total += 1

            if hasattr(module, "min_val"):

                if module.min_val is not None:

                    initialized += 1

                else:

                    uninitialized += 1

    print("=" * 50)
    print(f"Observers : {total}")
    print(f"Initialized : {initialized}")
    print(f"Uninitialized : {uninitialized}")
    print("=" * 50)
