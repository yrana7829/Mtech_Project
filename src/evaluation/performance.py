import time
import os
import torch


# -------------------------------
# Measure model size
# -------------------------------
def get_model_size(model_path):

    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return size_mb


# -------------------------------
# Measure latency
# -------------------------------
def measure_latency(model, dataloader, device, num_batches=20):

    model.eval()
    model.to(device)

    timings = []

    with torch.no_grad():

        # Warm-up
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            model(images)
            if i >= 5:
                break

        # Actual timing
        for i, (images, _) in enumerate(dataloader):

            images = images.to(device)

            start = time.time()
            model(images)
            end = time.time()

            timings.append(end - start)

            if i >= num_batches:
                break

    avg_latency = sum(timings) / len(timings)

    return avg_latency * 1000  # ms
