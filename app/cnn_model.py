import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssignmentCNN(nn.Module):
    def __init__(self):
        super(AssignmentCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )  # Input channels = 3, Output channels = 16

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )  # Input channels = 16, Output channels = 32

        # Assignment input is 64x64.
        # After two pooling layers: 64 -> 32 -> 16.
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # First block: Conv2D -> ReLU -> MaxPooling2D
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second block: Conv2D -> ReLU -> MaxPooling2D
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten -> Fully connected -> ReLU -> Fully connected
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
