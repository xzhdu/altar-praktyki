# Mistral OCR Test

Simple Python script for testing the Mistral OCR API (`mistral-ocr-latest`).

It extracts text and bounding boxes from image files (single images or directories of images), measures API response times, saves extracted markdown text and annotated visuals, and logs execution metadata to a single JSON file.

## Setup

1. **Install dependencies and sync environment:**
   Ensure [`uv`](https://docs.astral.sh/uv/) is installed, then run:
   ```bash
   uv sync
   ```

2. **Configure API Key:**
   Set the `MISTRAL_API_KEY` environment variable or create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env and set your MISTRAL_API_KEY
   ```

## Usage

You can run the script directly with `uv run` (or activate `.venv`):

### Single Image
```bash
uv run python mistral_ocr_test.py path/to/image.jpg -o ./output
```

### Directory of Images
```bash
uv run python mistral_ocr_test.py path/to/images_dir -o ./output
```

## Output Structure

The output directory contains:
```
output/
├── extracted_text/
│   └── <image_name>.md         # Extracted Markdown text
├── visuals/
│   └── <image_name>_annotated.<ext>  # Image with red/blue bounding boxes
└── results.json                # Run metadata (timestamps, response times, file names)
```
