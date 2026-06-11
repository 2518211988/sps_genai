from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import AssignmentCNN


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_cifar10.pth"

DEVICE = torch.device("cpu")

_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

_model = None
_classes = None


def load_classifier():
    global _model, _classes

    if _model is not None and _classes is not None:
        return _model, _classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = AssignmentCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model = model
    _classes = checkpoint["classes"]

    return _model, _classes


def predict_image(image: Image.Image):
    model, classes = load_classifier()

    image = image.convert("RGB")
    image_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    class_index = int(predicted_index.item())

    return {
        "class_index": class_index,
        "class_name": classes[class_index],
        "confidence": float(confidence.item()),
    }
