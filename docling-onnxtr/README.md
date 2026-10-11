# docling-onnxtr

Docling integration with the [OnnxTR](https://github.com/felixdittrich92/OnnxTR) OCR engine and [OpenCLIP](https://github.com/mlfoundations/open_clip) text embedding pipeline.

## Features

- **Automatic PDF OCR Detection**: Inspects PDF documents prior to conversion to determine if they contain programmatic digital text or require OCR.
  - Native digital PDFs bypass OCR for fast text extraction.
  - Scanned and image-based PDFs automatically route through the OnnxTR OCR engine.
- **OnnxTR OCR Engine**: Leverages lightweight, high-performance ONNX Runtime models for text detection and recognition.
- **Hybrid Semantic Chunking**: Uses Docling's `HybridChunker` to split documents into structured, contextual chunks respecting headings, paragraphs, and tables.
- **Vector Embeddings**: Generates multimodal vector embeddings for each text chunk using the `EmbeddingComponent` (OpenCLIP `xlm-roberta-base-ViT-B-32`).
- **Rich Metadata Export**: Exports chunks with comprehensive metadata ready for vector databases:
  - `file_id`: Document identifier / filename
  - `page_number` & `page_numbers`: Primary page and all spanned page numbers
  - `ocr_text`: Chunk text extracted via Docling / OCR
  - `is_ocr`: Flag indicating whether OCR was applied
  - `ocr_engine`: Name of the OCR engine (`OnnxTR` when OCR was used, `null` for native digital text)
  - `embedding`: Vector representation (512-dim)
  - `embedding_metadata`: Model name, execution time, and timestamp

## Installation

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
# Sync dependencies and create virtual environment:
uv sync

# For GPU acceleration:
uv sync --extra gpu
```

Alternatively, using a virtual environment with `uv pip`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# (or for GPU acceleration: uv pip install "docling-ocr-onnxtr[gpu]==0.2.1")
```

## Usage

### Command Line Interface (CLI)

Process a document, chunk it, generate text embeddings, and export chunks to JSON:

```bash
# Auto-detect whether OCR is needed (default mode):
uv run docling_onnxtr.py path/to/document.pdf

# Specify a custom output JSON file or output directory:
uv run docling_onnxtr.py path/to/document.pdf output.json
uv run docling_onnxtr.py path/to/document_scan.jpg ./output_dir/

# Force OCR execution regardless of digital text layer:
uv run docling_onnxtr.py path/to/document.pdf --force-ocr

# Disable OCR completely and rely exclusively on programmatic text extraction:
uv run docling_onnxtr.py path/to/document.pdf --no-ocr

# Custom digital text detection thresholds:
uv run docling_onnxtr.py path/to/document.pdf --min-chars 100 --min-ratio 0.8
```

*(Note: You can also activate the virtual environment via `source .venv/bin/activate` and run `python docling_onnxtr.py ...`).*

### CLI Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` | *required* | Path to input document (PDF or image). |
| `output_json_path` | `str` | `None` | Path to output JSON file or destination directory. |
| `--ocr-mode` | `auto` \| `force` \| `never` | `auto` | OCR strategy mode. |
| `--force-ocr` | flag | `False` | Shortcut for `--ocr-mode force`. |
| `--no-ocr` | flag | `False` | Shortcut for `--ocr-mode never`. |
| `--force-full-page-ocr` | flag | `False` | Force OCR across whole page instead of detected image regions. |
| `--min-chars` | `int` | `50` | Minimum non-whitespace characters per page to consider it having digital text. |
| `--min-ratio` | `float` | `0.5` | Ratio of pages with digital text required to bypass OCR. |

### Output JSON Format

The output JSON contains an array of chunk records:

```json
[
  {
    "chunk_id": 0,
    "file_id": "document.pdf",
    "page_number": 1,
    "page_numbers": [1],
    "ocr_text": "Sample extracted text from document...",
    "is_ocr": false,
    "ocr_engine": null,
    "embedding": [0.0123, -0.0456, ...],
    "embedding_metadata": {
      "file_name": "document.pdf",
      "model_name": "OpenCLIP xlm-roberta-base-ViT-B-32",
      "representation_type": "FixRes",
      "embedding_dim": 512,
      "execution_time_ms": 12.45,
      "timestamp": 1724581234.56
    }
  }
]
```

## Components

- [OpenCLIP Embedding Component](components/embedding/README.md) – Modular component for generating multimodal text and image embeddings.