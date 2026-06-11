from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.cnn_model import AssignmentCNN


torch.manual_seed(42)
np.random.seed(42)


CLASSES = np.array(
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)


def get_device():
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def main():
    batch_size = 32
    epochs = 1
    learning_rate = 0.0005

    device = get_device()
    print(f"Using device: {device}")

    # Assignment 2 model expects 64x64 RGB images.
    # CIFAR10 images are originally 32x32, so we resize them to 64x64.
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = AssignmentCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        model.train()

        for batch_number, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            if batch_number % 100 == 99:
                avg_loss = running_loss / (batch_number + 1)
                avg_acc = running_correct / running_total
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Batch {batch_number + 1}, "
                    f"avg loss: {avg_loss:.4f}, "
                    f"avg accuracy: {avg_acc:.3f}"
                )

    print("Finished Training")

    test_correct = 0
    test_total = 0

    model.eval()

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)

            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    checkpoint_path = models_dir / "cnn_cifar10.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES.tolist(),
            "input_size": [3, 64, 64],
            "test_accuracy": test_accuracy,
        },
        checkpoint_path,
    )

    print(f"Saved model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
