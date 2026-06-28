from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from app.gan_model import Generator, LATENT_DIM


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gan_mnist.pth"
)

DEVICE = torch.device("cpu")

_generator = None


def load_generator() -> Generator:
    """
    Load the trained Generator once and reuse it
    for later API requests.
    """
    global _generator

    if _generator is not None:
        return _generator

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GAN checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    latent_dim = checkpoint.get(
        "latent_dim",
        LATENT_DIM,
    )

    generator = Generator(
        latent_dim=latent_dim,
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.to(DEVICE)
    generator.eval()

    _generator = generator

    return _generator


def generate_digit_image() -> Image.Image:
    """
    Generate one random MNIST-style digit
    and return it as a PIL image.
    """
    generator = load_generator()

    noise = torch.randn(
        1,
        LATENT_DIM,
        device=DEVICE,
    )

    with torch.no_grad():
        generated_tensor = generator(noise)

    # Remove the batch dimension:
    # (1, 1, 28, 28) -> (1, 28, 28)
    generated_tensor = generated_tensor.squeeze(0).cpu()

    # Generator output uses Tanh and is in [-1, 1].
    # Convert it back to the display range [0, 1].
    generated_tensor = (generated_tensor + 1) / 2
    generated_tensor = generated_tensor.clamp(0, 1)

    return to_pil_image(generated_tensor)