import torch
import torch.nn as nn


LATENT_DIM = 100


class Generator(nn.Module):
    """
    Convert a random noise vector of shape (batch_size, 100)
    into a grayscale MNIST-style image of shape
    (batch_size, 1, 28, 28).
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        self.image_layers = nn.Sequential(
            # (B, 128, 7, 7) -> (B, 64, 14, 14)
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (B, 64, 14, 14) -> (B, 1, 28, 28)
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.fc(noise)
        x = x.view(noise.size(0), 128, 7, 7)
        return self.image_layers(x)


class Discriminator(nn.Module):
    """
    Classify a grayscale image as real or fake.

    Input shape:
        (batch_size, 1, 28, 28)

    Output shape:
        (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        self.image_layers = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 64, 14, 14)
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # (B, 64, 14, 14) -> (B, 128, 7, 7)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_layers(image)