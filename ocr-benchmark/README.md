# OCR Benchmark

A lightweight script that allows you to benchmark and compare different OCR engines (`RapidOCR`, `Tesseract`, `docTR`, `OnnxTR`) on the same files.

## Features

- **Multi-Engine Support**: Easily switch between `RapidOCR`, `Tesseract`, `docTR`, and `OnnxTR`.
- **Bounding Box Visualization**: Displays OCR detection bounding boxes and recognized text on a graphical window (using Matplotlib) to help evaluate quality.
- **Metric Evaluation**:
  - Automatically calculates **Word Error Rate (WER)** and **Character Error Rate (CER)** using `jiwer` if a ground truth file is provided.
  - Optional cross-checking against legacy edit-distance algorithms (`--cross-check`).
  - Repeat metric evaluation on existing test results (`repeat_existing_test` / `--repeat-test`) for files where ground truth is available without re-running OCR inference.
  - Measures execution time (seconds) and peak memory usage (RSS in MB).
- **Manual Scoring**: Prompts the user to input a manual score (1.0 to 5.0) for each analyzed document.
- **Batch Saving**: Save annotated images with drawn bounding boxes to a directory, and export all results to a JSON file.

## Prerequisites

Ensure you have [uv](https://docs.astral.sh/uv/) installed. If you don't have it installed:

```bash
# On Linux and macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sync project dependencies and set up the virtual environment:
```bash
uv sync
```
*(Note: Using Tesseract requires the `tesseract` binary to be installed on your system).*

## Usage

Run the benchmark using `uv run`:

### Basic Run
```bash
uv run ocr_benchmark.py -i /path/to/images -e rapidocr
# or using the entrypoint:
uv run ocr-benchmark -i /path/to/images -e rapidocr
```

### With Ground Truth & Output JSON
To compute error rates and save the benchmark results:
```bash
uv run ocr_benchmark.py -i /path/to/images -e onnxtr -g ground_truth.json -o results.json
```

### Repeat Existing Test with Ground Truth
To re-evaluate metrics on a previously saved results JSON file, updating WER and CER for files with ground truth reference text while keeping all files from the original test:
```bash
uv run ocr_benchmark.py --repeat-test results.json -g ground_truth.json -o updated_results.json
```
*(Add `--only-ground-truth` if you want the output to only include files that have reference text).*

### Filter Benchmark for Files with Ground Truth Only
To run OCR inference only on images that have an entry in the ground truth file:
```bash
uv run ocr_benchmark.py -i /path/to/images -e onnxtr -g ground_truth.json --only-ground-truth -o results.json
```

### Headless Mode & Saving Visualizations
To run without opening GUI windows and save the annotated bounding box images:
```bash
uv run ocr_benchmark.py -i /path/to/images -e onnxtr --no-display --save-visual-dir ./output_visuals
```

## CLI Options

| Argument | Description |
|---|---|
| `-i`, `--input` | Path to input image file or directory containing images (required unless `--repeat-test` is used). |
| `-r`, `--repeat-test` | Path to existing benchmark results JSON file to repeat metric evaluation on files with ground truth. |
| `-e`, `--engine` | OCR engine to use: `rapidocr` (default), `tesseract`, `doctr`, or `onnxtr`. |
| `-g`, `--ground-truth` | Path to a ground truth JSON file mapping filenames to expected reference text. |
| `-o`, `--output` | Path to save the benchmark metrics in JSON format. |
| `--only-ground-truth` | Only run benchmark on files where ground truth reference text is available. |
| `--no-display` | Disable displaying annotated bounding boxes in a GUI window. |
| `--save-visual-dir` | Directory to save annotated images with bounding boxes. |
| `--cross-check` | Cross-check jiwer CER and WER results against legacy edit-distance algorithms. |
