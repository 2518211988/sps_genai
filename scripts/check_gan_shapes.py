import torch

from app.gan_model import Discriminator, Generator, LATENT_DIM


def main():
    batch_size = 4

    generator = Generator()
    discriminator = Discriminator()

    noise = torch.randn(batch_size, LATENT_DIM)

    with torch.no_grad():
        fake_images = generator(noise)
        predictions = discriminator(fake_images)

    print("Noise shape:         ", noise.shape)
    print("Generated image shape:", fake_images.shape)
    print("Prediction shape:     ", predictions.shape)
    print(
        "Generated value range:",
        float(fake_images.min()),
        "to",
        float(fake_images.max()),
    )
    print(
        "Prediction range:     ",
        float(predictions.min()),
        "to",
        float(predictions.max()),
    )

    assert noise.shape == (batch_size, 100)
    assert fake_images.shape == (batch_size, 1, 28, 28)
    assert predictions.shape == (batch_size, 1)

    assert fake_images.min() >= -1
    assert fake_images.max() <= 1

    assert predictions.min() >= 0
    assert predictions.max() <= 1

    print("\nAll GAN architecture checks passed.")


if __name__ == "__main__":
    main()