# APANPS5560 Applied Generative AI Assignments

This repository contains the reusable FastAPI application developed across multiple APANPS5560 Applied Generative AI assignments.

## Included Assignments

### Assignment 1: Text Generation and Word Embeddings

Assignment 1 introduced the reusable FastAPI application with:

- A bigram text generation endpoint
- A spaCy word embedding endpoint
- Docker deployment support

### Assignment 2: CNN Image Classification

Assignment 2 added:

- A PyTorch convolutional neural network
- CIFAR-10 model training
- A saved CNN checkpoint
- An image classification API endpoint

### Assignment 3: MNIST GAN Image Generation

Assignment 3 adds:

- A PyTorch Generator that converts a 100-dimensional noise vector into a 28×28 grayscale image
- A PyTorch Discriminator that distinguishes real and generated images
- MNIST GAN training and architecture validation scripts
- A trained GAN checkpoint
- An API endpoint that returns a generated digit as a PNG image

## Requirements

- Python 3.12
- uv
- Docker, optional

## Install Dependencies

```bash
uv sync
uv run python -m spacy download en_core_web_md
````

## Run the API Locally

```bash
uv run fastapi dev app/main.py
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Root

```http
GET /
```

Expected response:

```json
{"Hello":"World"}
```

### Generate Text

```http
POST /generate
```

Example request:

```json
{
  "start_word": "the",
  "length": 10
}
```

### Word Embedding

```http
POST /embedding
```

Example request:

```json
{
  "input_word": "dog"
}
```

The response includes the input word, embedding dimension, and embedding vector.

### Classify an Image

```http
POST /classify-image
```

Upload an image file using the FastAPI documentation page. The endpoint returns the predicted CIFAR-10 class and confidence score.

### Generate a Handwritten Digit

```http
GET /generate-digit
```

The endpoint uses the trained GAN Generator and returns a randomly generated 28×28 grayscale PNG image.

## Assignment 2: Train the CNN

```bash
uv run python -m scripts.train_cnn
```

The trained CNN checkpoint is saved to:

```text
models/cnn_cifar10.pth
```

## Assignment 3: Validate the GAN Architecture

```bash
uv run python -m scripts.check_gan_shapes
```

This checks that:

* The noise input has shape `(batch_size, 100)`
* The Generator output has shape `(batch_size, 1, 28, 28)`
* The Discriminator output has shape `(batch_size, 1)`

## Assignment 3: Train the GAN

```bash
uv run python -m scripts.train_gan
```

The trained GAN checkpoint is saved to:

```text
models/gan_mnist.pth
```

Generated images from each training epoch are saved locally under `samples/`.

## Run with Docker

Build the Docker image:

```bash
docker build -t sps-genai-assignment3 .
```

Run the container:

```bash
docker run --rm -p 8010:80 sps-genai-assignment3
```

Open the API documentation:

```text
http://127.0.0.1:8010/docs
```

Generate a digit from the Docker deployment:

```text
http://127.0.0.1:8010/generate-digit
```

## Project Structure

```text
app/
  bigram_model.py
  cnn_model.py
  embedding_model.py
  gan_generator.py
  gan_model.py
  image_classifier.py
  main.py

models/
  cnn_cifar10.pth
  gan_mnist.pth

scripts/
  check_gan_shapes.py
  train_cnn.py
  train_gan.py
```

## Course

APANPS5560_D01_2026_2 — Applied Generative AI
Columbia University School of Professional Studies
