import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling_ocr_onnxtr import OnnxtrOcrOptions

from component import EmbeddingComponent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DoclingOnnxTR")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}


@dataclass
class PdfOcrAnalysisResult:
    """Result of analyzing a document to determine if OCR is required."""

    needs_ocr: bool
    reason: str
    total_pages: int = 0
    text_pages_count: int = 0
    total_chars: int = 0
    avg_chars_per_page: float = 0.0
    text_page_ratio: float = 0.0
    ocr_mode: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def should_use_ocr(
    input_path: Union[str, Path],
    ocr_mode: str = "auto",
    min_chars_per_page: int = 50,
    min_text_page_ratio: float = 0.5,
) -> PdfOcrAnalysisResult:
    # Użycie resolved path, aby uniknąć błędów ze znakami specjalnymi na Windows
    path = Path(input_path).resolve()
    ocr_mode_lower = (ocr_mode or "auto").strip().lower()

    if ocr_mode_lower in ("force", "always", "true", "1"):
        return PdfOcrAnalysisResult(
            needs_ocr=True,
            reason="OCR forced by user configuration (ocr_mode='force')",
            ocr_mode=ocr_mode_lower,
        )
    elif ocr_mode_lower in ("never", "disabled", "none", "false", "0"):
        return PdfOcrAnalysisResult(
            needs_ocr=False,
            reason="OCR disabled by user configuration (ocr_mode='never')",
            ocr_mode=ocr_mode_lower,
        )

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return PdfOcrAnalysisResult(
            needs_ocr=True,
            reason=f"Image file format ({path.suffix}) requires OCR",
            ocr_mode=ocr_mode_lower,
        )

    if path.suffix.lower() == ".pdf":
        pdf = None
        try:
            import pypdfium2 as pdfium

            # Otwarcie dokumentu PDF
            pdf = pdfium.PdfDocument(str(path))
            total_pages = len(pdf)

            if total_pages == 0:
                return PdfOcrAnalysisResult(
                    needs_ocr=False,
                    reason="Document has 0 pages (empty PDF)",
                    total_pages=0,
                    ocr_mode=ocr_mode_lower,
                )

            pages_with_text = 0
            total_chars = 0

            for i in range(total_pages):
                page = pdf[i]
                textpage = page.get_textpage()
                text = textpage.get_text_bounded()
                non_ws_count = len(text.strip())
                total_chars += non_ws_count
                if non_ws_count >= min_chars_per_page:
                    pages_with_text += 1

            text_page_ratio = pages_with_text / total_pages if total_pages > 0 else 0.0
            avg_chars = total_chars / total_pages if total_pages > 0 else 0.0

            # KLUCZOWE: Zamykamy plik od razu po odczytaniu danych!
            pdf.close()
            pdf = None

            if text_page_ratio >= min_text_page_ratio:
                return PdfOcrAnalysisResult(
                    needs_ocr=False,
                    reason=(
                        f"Digital text layer detected: {pages_with_text}/{total_pages} pages "
                        f"({text_page_ratio * 100:.1f}%) have >= {min_chars_per_page} chars. OCR bypassed."
                    ),
                    total_pages=total_pages,
                    text_pages_count=pages_with_text,
                    total_chars=total_chars,
                    avg_chars_per_page=avg_chars,
                    text_page_ratio=text_page_ratio,
                    ocr_mode=ocr_mode_lower,
                )
            else:
                return PdfOcrAnalysisResult(
                    needs_ocr=True,
                    reason=f"Scanned / low-text PDF detected. OCR is required.",
                    total_pages=total_pages,
                    text_pages_count=pages_with_text,
                    total_chars=total_chars,
                    avg_chars_per_page=avg_chars,
                    text_page_ratio=text_page_ratio,
                    ocr_mode=ocr_mode_lower,
                )
        except Exception as exc:
            logger.warning(f"Inspection failed ({exc}), falling back to OCR.")
            return PdfOcrAnalysisResult(
                needs_ocr=True,
                reason=f"Inspection failed ({exc}), falling back to OCR",
                ocr_mode=ocr_mode_lower,
            )
        finally:
            # Gwarancja zwolnienia blokady pliku w pamięci
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass

    return PdfOcrAnalysisResult(
        needs_ocr=True,
        reason=f"Unsupported format ({path.suffix}), defaulting to OCR",
        ocr_mode=ocr_mode_lower,
    )

def create_document_converter(
    do_ocr: bool = True,
    force_full_page_ocr: bool = False,
) -> DocumentConverter:
    """
    Create and configure a Docling DocumentConverter.

    If do_ocr is True, configures the OnnxTR OCR engine.
    If do_ocr is False, disables OCR and relies on native programmatic text extraction.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr

    if do_ocr:
        pipeline_options.allow_external_plugins = True
        pipeline_options.ocr_options = OnnxtrOcrOptions(
            force_full_page_ocr=force_full_page_ocr
        )

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }

    return DocumentConverter(format_options=format_options)


def create_chunker() -> HybridChunker:
    """Create and configure a Docling HybridChunker."""
    return HybridChunker()


def chunk_and_embed_document(
    doc: DoclingDocument,
    file_id: str,
    embedder: Optional[EmbeddingComponent] = None,
    chunker: Optional[HybridChunker] = None,
    ocr_engine_name: Optional[str] = "OnnxTR",
    is_ocr: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument using HybridChunker and generate embeddings for each chunk.

    Each chunk contains:
    - chunk_id: index of the chunk
    - file_id: identifier of the source file
    - page_number: primary (first) page number of the chunk
    - page_numbers: list of all page numbers spanned by the chunk
    - ocr_text: the chunk text extracted via OCR / Docling
    - is_ocr: flag indicating whether the text was processed by OCR
    - ocr_engine: name of the OCR engine used (e.g., 'OnnxTR'), or None if native text was extracted
    - embedding: vector embedding generated by EmbeddingComponent
    - embedding_metadata: metadata about the embedding model and execution timing
    """
    if chunker is None:
        chunker = create_chunker()

    if embedder is None:
        embedder = EmbeddingComponent()

    raw_chunks = list(chunker.chunk(doc))
    embedded_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(raw_chunks):
        chunk_text = chunk.text
        if not chunk_text or not chunk_text.strip():
            continue

        # Extract page numbers from provenance items in doc_items
        page_set = set()
        for item in getattr(chunk.meta, "doc_items", []):
            for prov in getattr(item, "prov", []):
                if hasattr(prov, "page_no") and prov.page_no is not None:
                    page_set.add(int(prov.page_no))

        page_numbers = sorted(list(page_set))
        if not page_numbers and doc.pages:
            page_numbers = [min(doc.pages.keys())]
        elif not page_numbers:
            page_numbers = [1]

        primary_page = page_numbers[0]

        # Generate embedding and time it
        start_time = time.time()
        embedding_vector = embedder.embed_text(chunk_text)
        exec_time_ms = (time.time() - start_time) * 1000.0

        embedding_meta = embedder.get_metadata(
            file_path=file_id,
            execution_time_ms=exec_time_ms,
        )

        chunk_record: Dict[str, Any] = {
            "chunk_id": idx,
            "file_id": file_id,
            "page_number": primary_page,
            "page_numbers": page_numbers,
            "ocr_text": chunk_text,
            "is_ocr": is_ocr,
            "ocr_engine": ocr_engine_name if is_ocr else None,
            "embedding": embedding_vector,
            "embedding_metadata": embedding_meta,
        }
        embedded_chunks.append(chunk_record)

    return embedded_chunks


def save_chunks_to_json(
    chunks: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Save the list of embedded chunks to a JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return out_file


def process_document(
    input_path: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
    ocr_mode: str = "auto",
    min_chars_per_page: int = 50,
    min_text_page_ratio: float = 0.5,
    force_full_page_ocr: bool = False,
    embedder: Optional[EmbeddingComponent] = None,
) -> Tuple[DoclingDocument, List[Dict[str, Any]]]:

    # Zawsze przekazujemy absolutną, rozwiązaną ścieżkę
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Document not found: {input_file}")

    file_id = input_file.name

    ocr_analysis = should_use_ocr(
        input_path=input_file,
        ocr_mode=ocr_mode,
        min_chars_per_page=min_chars_per_page,
        min_text_page_ratio=min_text_page_ratio,
    )
    do_ocr = ocr_analysis.needs_ocr

    converter = create_document_converter(
        do_ocr=do_ocr,
        force_full_page_ocr=force_full_page_ocr,
    )
    
    # Przekazujemy czytelną dla C++ absolutną ścieżkę w postaci stringa
    conv_res = converter.convert(str(input_file))
    doc: DoclingDocument = conv_res.document

    chunks = chunk_and_embed_document(
        doc=doc,
        file_id=file_id,
        embedder=embedder,
        ocr_engine_name="OnnxTR" if do_ocr else None,
        is_ocr=do_ocr,
    )

    if output_json_path is not None:
        out_file = Path(output_json_path)
        if out_file.is_dir():
            out_file = out_file / f"{input_file.stem}.chunks.json"
        save_chunks_to_json(chunks, out_file)

    return doc, chunks
"""
    Process a document with Docling:
    1. Determine whether OCR is required (via auto-detection or specified mode).
    2. Convert the document using Docling (with OnnxTR OCR if required, or native text extraction).
    3. Chunk the document using HybridChunker and generate text embeddings.
    4. Save the embedded chunks with accurate OCR metadata to a JSON file.
"""
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process documents with Docling, automatic OnnxTR OCR detection, and OpenCLIP embeddings."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the document (PDF or image) to process.",
    )
    parser.add_argument(
        "output_json_path",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to output JSON file or directory (default: <input_stem>.chunks.json).",
    )
    parser.add_argument(
        "--ocr-mode",
        type=str,
        choices=["auto", "force", "never"],
        default="auto",
        help="OCR mode: 'auto' (inspect PDF text layer), 'force' (always run OCR), 'never' (disable OCR). Default: 'auto'.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR processing regardless of digital text layer (shortcut for --ocr-mode force).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR completely (shortcut for --ocr-mode never).",
    )
    parser.add_argument(
        "--force-full-page-ocr",
        action="store_true",
        help="Force OCR on the entire page area instead of only detected image regions when OCR is active.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum characters per page threshold for digital text detection (default: 50).",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of text-containing pages to classify PDF as digital (default: 0.5).",
    )

    args = parser.parse_args()

    doc_path = Path(args.input_path)
    out_json_path = (
        Path(args.output_json_path)
        if args.output_json_path
        else doc_path.with_name(f"{doc_path.stem}.chunks.json")
    )

    # Determine resolved ocr_mode
    ocr_mode = args.ocr_mode
    if args.force_ocr:
        ocr_mode = "force"
    elif args.no_ocr:
        ocr_mode = "never"

    print(f"Processing '{doc_path}' (ocr_mode='{ocr_mode}')...")
    doc, chunks = process_document(
        input_path=doc_path,
        output_json_path=out_json_path,
        ocr_mode=ocr_mode,
        min_chars_per_page=args.min_chars,
        min_text_page_ratio=args.min_ratio,
        force_full_page_ocr=args.force_full_page_ocr,
    )

    print(f"Successfully processed: {doc.name}")
    print(f"Total pages: {len(doc.pages)}")
    print(f"Total chunks created & embedded: {len(chunks)}")
    print(f"Saved chunks to: {out_json_path}")


if __name__ == "__main__":
    main()