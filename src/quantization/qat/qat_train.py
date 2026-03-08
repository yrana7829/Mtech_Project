import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq

from .qat_prepare import prepare_qat_model


def run_qat_training(model, train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = prepare_qat_model(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    model.eval()

    quantized_model = tq.convert(model.eval(), inplace=False)

    return quantized_model
