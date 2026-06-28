from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Discriminator, Generator, LATENT_DIM


# Make the experiment more reproducible.
torch.manual_seed(42)


def get_device() -> torch.device:
    """
    Use Apple Silicon GPU when available,
    otherwise use CUDA or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    # -------------------------
    # 1. Training configuration
    # -------------------------
    batch_size = 128
    epochs = 5
    learning_rate = 0.0002

    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # 2. Load the MNIST dataset
    # -------------------------
    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            # MNIST normally has pixel values from 0 to 1.
            # Convert them to -1 to 1 to match Generator's Tanh output.
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    # -------------------------
    # 3. Create the two models
    # -------------------------
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Discriminator outputs probabilities from 0 to 1,
    # so Binary Cross Entropy is used.
    criterion = nn.BCELoss()

    # GAN uses one optimizer for each network.
    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    # Use the same fixed noise after every epoch.
    # This lets us visually compare whether the model improves.
    fixed_noise = torch.randn(16, LATENT_DIM, device=device)

    samples_dir = Path("samples")
    models_dir = Path("models")

    samples_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # -------------------------
    # 4. GAN training loop
    # -------------------------
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        discriminator_loss_total = 0.0
        generator_loss_total = 0.0

        for batch_number, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            # ==========================================
            # A. Train the Discriminator
            # ==========================================
            discriminator_optimizer.zero_grad()

            # Real images should be predicted as 1.
            real_predictions = discriminator(real_images)
            real_targets = torch.ones_like(real_predictions)

            real_loss = criterion(
                real_predictions,
                real_targets,
            )

            # Generate fake images.
            noise = torch.randn(
                current_batch_size,
                LATENT_DIM,
                device=device,
            )

            fake_images = generator(noise)

            # Fake images should be predicted as 0.
            #
            # detach() prevents this discriminator update
            # from modifying the Generator.
            fake_predictions = discriminator(fake_images.detach())
            fake_targets = torch.zeros_like(fake_predictions)

            fake_loss = criterion(
                fake_predictions,
                fake_targets,
            )

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # ==========================================
            # B. Train the Generator
            # ==========================================
            generator_optimizer.zero_grad()

            noise = torch.randn(
                current_batch_size,
                LATENT_DIM,
                device=device,
            )

            generated_images = generator(noise)
            generated_predictions = discriminator(generated_images)

            # Generator wants fake images to be judged as real.
            generator_targets = torch.ones_like(generated_predictions)

            generator_loss = criterion(
                generated_predictions,
                generator_targets,
            )

            generator_loss.backward()
            generator_optimizer.step()

            discriminator_loss_total += discriminator_loss.item()
            generator_loss_total += generator_loss.item()

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Batch {batch_number}/{len(train_loader)}, "
                    f"D loss: {discriminator_loss.item():.4f}, "
                    f"G loss: {generator_loss.item():.4f}"
                )

        average_discriminator_loss = (
            discriminator_loss_total / len(train_loader)
        )
        average_generator_loss = (
            generator_loss_total / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1} finished | "
            f"Average D loss: {average_discriminator_loss:.4f} | "
            f"Average G loss: {average_generator_loss:.4f}"
        )

        # ------------------------------------------
        # Save images generated from the same noise
        # ------------------------------------------
        generator.eval()

        with torch.no_grad():
            sample_images = generator(fixed_noise).cpu()

        sample_path = samples_dir / f"gan_epoch_{epoch + 1}.png"

        save_image(
            sample_images,
            sample_path,
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )

        print(f"Saved sample image to {sample_path}")

    # -------------------------
    # 5. Save trained weights
    # -------------------------
    checkpoint_path = models_dir / "gan_mnist.pth"

    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "latent_dim": LATENT_DIM,
            "image_shape": [1, 28, 28],
            "epochs": epochs,
        },
        checkpoint_path,
    )

    print("Finished GAN training.")
    print(f"Saved model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()