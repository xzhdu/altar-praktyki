"""Mistral OCR extraction script.

Extracts text and visual bounding boxes from images using the Mistral OCR API.
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageDraw
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def get_mistral_client(api_key: Optional[str] = None) -> Mistral:
    """Initialize and return a Mistral client using the provided key or environment variable."""
    load_dotenv()
    key = api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError(
            "MISTRAL_API_KEY is not set. Please set the environment variable or create a .env file."
        )
    return Mistral(api_key=key)


def encode_image_to_base64(image_path: Path) -> str:
    """Read a local image file and convert it into a base64 data URI."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    with open(image_path, "rb") as f:
        encoded_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_data}"


def process_ocr_request(
    client: Mistral,
    image_path: Path,
    model: str = "mistral-ocr-latest",
) -> Tuple[object, float]:
    """Send an image to Mistral OCR API, measure response time, and return response with elapsed time."""
    image_data_uri = encode_image_to_base64(image_path)

    start_time = time.perf_counter()
    response = client.ocr.process(
        model=model,
        document={
            "type": "image_url",
            "image_url": image_data_uri,
        },
        include_image_base64=True,
        include_blocks=True,
    )
    elapsed_time = time.perf_counter() - start_time

    print(f"[{image_path.name}] OCR response received in {elapsed_time:.2f} seconds")
    return response, elapsed_time


def save_extracted_markdown(
    ocr_response: object,
    output_dir: Path,
    base_filename: str,
) -> Path:
    """Save extracted markdown text from OCR response into extracted_text/ directory."""
    text_dir = output_dir / "extracted_text"
    text_dir.mkdir(parents=True, exist_ok=True)

    md_file_path = text_dir / f"{base_filename}.md"

    markdown_parts = []
    pages = getattr(ocr_response, "pages", []) or []
    for page in pages:
        md = getattr(page, "markdown", "")
        if md:
            markdown_parts.append(md)

    full_markdown = "\n\n".join(markdown_parts)
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    return md_file_path


def save_visuals_with_bounding_boxes(
    ocr_response: object,
    original_image_path: Path,
    output_dir: Path,
    base_filename: str,
) -> Path:
    """Draw bounding boxes from OCR response on the original image and save to visuals/ directory."""
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    suffix = original_image_path.suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        suffix = ".png"
    visual_file_path = visuals_dir / f"{base_filename}_annotated{suffix}"

    with Image.open(original_image_path) as img:
        annotated_img = img.convert("RGB")
        draw = ImageDraw.Draw(annotated_img)
        img_w, img_h = annotated_img.size

        pages = getattr(ocr_response, "pages", []) or []
        for page in pages:
            dims = getattr(page, "dimensions", None)
            page_w = getattr(dims, "width", None) if dims else None
            page_h = getattr(dims, "height", None) if dims else None

            scale_x = (img_w / page_w) if page_w else 1.0
            scale_y = (img_h / page_h) if page_h else 1.0

            # Draw block bounding boxes
            blocks = getattr(page, "blocks", []) or []
            for block in blocks:
                x0 = getattr(block, "top_left_x", None)
                y0 = getattr(block, "top_left_y", None)
                x1 = getattr(block, "bottom_right_x", None)
                y1 = getattr(block, "bottom_right_y", None)

                if None not in (x0, y0, x1, y1):
                    if 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0:
                        coords = [x0 * img_w, y0 * img_h, x1 * img_w, y1 * img_h]
                    else:
                        coords = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                    draw.rectangle(coords, outline="red", width=2)

            # Draw image block bounding boxes if any
            images = getattr(page, "images", []) or []
            for img_block in images:
                x0 = getattr(img_block, "top_left_x", None)
                y0 = getattr(img_block, "top_left_y", None)
                x1 = getattr(img_block, "bottom_right_x", None)
                y1 = getattr(img_block, "bottom_right_y", None)

                if None not in (x0, y0, x1, y1):
                    if 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0:
                        coords = [x0 * img_w, y0 * img_h, x1 * img_w, y1 * img_h]
                    else:
                        coords = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                    draw.rectangle(coords, outline="blue", width=2)

        annotated_img.save(visual_file_path)

    return visual_file_path


def process_image(
    client: Mistral,
    image_path: Path,
    output_dir: Path,
) -> dict:
    """Process a single image, save its text and visuals, and return a metadata dictionary."""
    ocr_resp, response_time = process_ocr_request(client, image_path)
    base_name = image_path.stem

    saved_md_path = save_extracted_markdown(ocr_resp, output_dir, base_name)
    saved_vis_path = save_visuals_with_bounding_boxes(ocr_resp, image_path, output_dir, base_name)

    return {
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "file_name": image_path.name,
        "response_time_seconds": round(response_time, 3),
        "markdown_file": str(saved_md_path.relative_to(output_dir)),
        "visual_file": str(saved_vis_path.relative_to(output_dir)),
    }


def save_metadata_json(output_dir: Path, records: List[dict]) -> Path:
    """Save or append metadata records into a single JSON file inside the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"

    existing_records = []
    if json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_records = json.load(f)
        except Exception:
            existing_records = []

    existing_records.extend(records)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2)

    return json_path


def collect_images(input_path: Path) -> List[Path]:
    """Collect image paths from a single file or a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Warning: File '{input_path.name}' does not have a standard image extension.")
        return [input_path]
    elif input_path.is_dir():
        return sorted([
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
    return []


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract text and bounding box visuals from images using Mistral OCR API."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("./output"),
        help="Directory to save outputs and results.json (default: ./output).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_path.exists():
        print(f"Error: Input path '{args.input_path}' does not exist.")
        sys.exit(1)

    images = collect_images(args.input_path)
    if not images:
        print(f"Error: No image files found in '{args.input_path}'.")
        sys.exit(1)

    try:
        mistral_client = get_mistral_client()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process. Output directory: {args.output_dir}")
    records = []

    for img_path in images:
        try:
            record = process_image(mistral_client, img_path, args.output_dir)
            records.append(record)
        except Exception as e:
            print(f"[{img_path.name}] Error during OCR processing: {e}")

    if records:
        json_file = save_metadata_json(args.output_dir, records)
        print(f"\nCompleted {len(records)}/{len(images)} images.")
        print(f"Saved metadata log to: {json_file}")


if __name__ == "__main__":
    main()