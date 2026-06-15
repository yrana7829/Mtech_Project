import os
import torch
from tqdm import tqdm


class QATTrainer:

    def __init__(self, model, train_loader, val_loader, device, lr=1e-5):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        train_acc = correct / total

        return total_loss / len(self.train_loader), train_acc

    def validate(self):

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.val_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def train(self, epochs=15, save_path="results/checkpoints/qat_model.pth"):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        best_val_acc = 0

        for epoch in range(epochs):

            # Freeze BN stats
            if epoch == 7:
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                print("BatchNorm frozen")

            # Freeze observers
            if epoch == 8:
                self.model.apply(torch.ao.quantization.disable_observer)
                print("Observers frozen")

            train_loss, train_acc = self.train_epoch()

            val_acc = self.validate()

            self.scheduler.step()

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(self.model.state_dict(), save_path)

                print("Best QAT model saved")
