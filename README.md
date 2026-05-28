# Assignment 1: FastAPI Word Embedding API

This project implements a FastAPI application for APANPS5560 Assignment 1.

It includes:

- A bigram text generation endpoint from the Module 2 class activity
- A spaCy word embedding endpoint based on the Module 1 word embeddings activity
- Optional Docker deployment support

## Requirements

- Python 3.12
- uv
- Docker optional

## Run Locally

Install dependencies:

```bash
uv sync
uv run python -m spacy download en_core_web_md
````

Start the API:

```bash
uv run fastapi dev app/main.py
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

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

Expected response includes:

```json
{
  "input_word": "dog",
  "dimension": 300,
  "embedding": [...]
}
```

## Run with Docker

Build the image:

```bash
docker build -t sps-genai .
```

Run the container:

```bash
docker run --rm -p 8010:80 sps-genai
```

Open:

```text
http://127.0.0.1:8010/docs
```

