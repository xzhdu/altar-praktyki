import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import docling_szymon
from component import EmbeddingComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Indexer")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS.union({".pdf"})

# Nazwy dwóch niezależnych kolekcji
COLLECTION_DOCUMENTS = "documents_ocr"
COLLECTION_IMAGES = "images_visual"


def process_single_file(
    file_path: Path, 
    embedder: EmbeddingComponent,
    min_word_count: int = 15
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Kieruje plik do odpowiedniej ścieżki i zwraca chunki wraz z nazwą docelowej kolekcji.
    """
    ext = file_path.suffix.lower()
    
    # 1. PDF -> Kolekcja dokumentów
    if ext == ".pdf":
        _, chunks = docling_szymon.process_document(
            input_path=file_path, 
            ocr_mode="auto", 
            embedder=embedder
        )
        return chunks, COLLECTION_DOCUMENTS

    # 2. Obrazy (JPG, PNG itp.)
    if ext in IMAGE_EXTENSIONS:
        try:
            _, chunks = docling_szymon.process_document(
                input_path=file_path, 
                ocr_mode="force", 
                embedder=embedder
            )
            
            total_words = sum(len(c.get("ocr_text", "").split()) for c in chunks)
            
            # Jeśli znaleziono >= 15 słów -> Ścieżka Dokumentów A4/Skanów
            if chunks and total_words >= min_word_count:
                logger.info(f" Dokument OCR ({total_words} słów): {file_path.name} -> {COLLECTION_DOCUMENTS}")
                return chunks, COLLECTION_DOCUMENTS
            else:
                logger.info(f" Czysty obraz ({total_words} słów): {file_path.name} -> {COLLECTION_IMAGES}")
        except Exception as e:
            logger.debug(f"Błąd OCR dla {file_path.name}: {e}")

        # Poniżej 15 słów -> Ścieżka Czystego Obrazu (OpenCLIP Visual)
        start_time = time.perf_counter()
        vector = embedder.embed_image(file_path)
        exec_time_ms = (time.perf_counter() - start_time) * 1000
        
        image_record = [{
            "chunk_id": 0,
            "file_id": file_path.name,
            "page_number": 1,
            "page_numbers": [1],
            "ocr_text": "",
            "is_ocr": False,
            "ocr_engine": None,
            "embedding": vector,
            "embedding_metadata": embedder.get_metadata(
                file_path=str(file_path),
                execution_time_ms=exec_time_ms
            )
        }]
        return image_record, COLLECTION_IMAGES

    return [], COLLECTION_DOCUMENTS


def index_directory(
    input_dir: Path,
    qdrant_client: QdrantClient,
    embedder: EmbeddingComponent,
    batch_size: int = 50
) -> Dict[str, int]:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Katalog '{input_dir}' nie istnieje.")

    embedding_dim = embedder.embedding_dim

    # Inicjalizacja obu osobnych kolekcji w Qdrant
    for coll in [COLLECTION_DOCUMENTS, COLLECTION_IMAGES]:
        if not qdrant_client.collection_exists(coll):
            logger.info(f"Tworzenie kolekcji '{coll}' (wymiar: {embedding_dim})...")
            qdrant_client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )

    all_files = [f for f in input_path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    logger.info(f"Znaleziono {len(all_files)} plików do zindeksowania.")

    buffers = {COLLECTION_DOCUMENTS: [], COLLECTION_IMAGES: []}
    totals = {COLLECTION_DOCUMENTS: 0, COLLECTION_IMAGES: 0}

    for file_path in all_files:
        logger.info(f"Przetwarzanie: {file_path.name}")
        try:
            records, target_collection = process_single_file(file_path, embedder)
        except Exception as e:
            logger.error(f"Błąd przetwarzania pliku {file_path.name}: {e}")
            continue

        for record in records:
            file_id = record.get("file_id", "unknown")
            chunk_id = record.get("chunk_id", 0)
            
            # Deterministyczne ID zapobiegające duplikacji
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{target_collection}_{file_id}_chunk_{chunk_id}"))
            vector = record.get("embedding")

            if not vector:
                continue

            payload = {k: v for k, v in record.items() if k != "embedding"}
            buffers[target_collection].append(PointStruct(id=point_id, vector=vector, payload=payload))

            if len(buffers[target_collection]) >= batch_size:
                qdrant_client.upsert(collection_name=target_collection, points=buffers[target_collection])
                totals[target_collection] += len(buffers[target_collection])
                buffers[target_collection].clear()

    # Czyszczenie pozostałych punktów z buforów
    for coll, buf in buffers.items():
        if buf:
            qdrant_client.upsert(collection_name=coll, points=buf)
            totals[coll] += len(buf)

    logger.info(f"Zakończono! Zindeksowano -> Dokumenty (OCR): {totals[COLLECTION_DOCUMENTS]}, Obrazy: {totals[COLLECTION_IMAGES]}")
    return totals


if __name__ == "__main__":
    client = QdrantClient(path=r"C:\Users\mateu\Desktop\Praktyki_altar\qdrant_db")
    embedder_comp = EmbeddingComponent(allow_cloud=False)
    data_directory = Path(r"C:\Users\mateu\Desktop\Zbior_danych")

    if data_directory.exists():
        index_directory(data_directory, client, embedder_comp)
    else:
        logger.error(f"Katalog {data_directory} nie istnieje.")

    client.close()