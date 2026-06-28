from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding
from app.gan_generator import generate_digit_image
from app.image_classifier import predict_image


app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    input_word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.input_word)
    return {
        "input_word": request.input_word,
        "dimension": len(embedding),
        "embedding": embedding,
    }


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        prediction = predict_image(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "filename": file.filename,
        **prediction,
    }

@app.get(
    "/generate-digit",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {}
            },
            "description": (
                "A randomly generated 28×28 "
                "grayscale MNIST-style digit."
            ),
        }
    },
)
def generate_digit():
    try:
        image = generate_digit_image()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")

    return Response(
        content=image_buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": (
                "inline; filename=generated_digit.png"
            )
        },
    )