import logging
from pathlib import Path
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from component import EmbeddingComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SearchEngine")

COLLECTION_DOCUMENTS = "documents_ocr"
COLLECTION_IMAGES = "images_visual"


def search_multimodal(
    query_text: str,
    qdrant_client: QdrantClient,
    embedder: EmbeddingComponent,
    limit_per_category: int = 3,
    doc_score_threshold: float = None,
    img_score_threshold: float = 0.15
) -> Dict[str, List[Dict[str, Any]]]:

    logger.info(f"Generowanie wektora zapytania dla: '{query_text}'")
    
    # Przeznaczenie osobnych wektorów dla przestrzeni tekstowej i wizualnej
    query_vector_text = embedder.embed_text(query_text)
    query_vector_image = embedder.embed_text(f"zdjęcie przedstawiające {query_text}")

    # 1. Przeszukiwanie bazy DOKUMENTÓW (OCR)
    doc_response = qdrant_client.query_points(
        collection_name=COLLECTION_DOCUMENTS,
        query=query_vector_text,
        limit=limit_per_category * 3,
        score_threshold=doc_score_threshold
    ) if qdrant_client.collection_exists(COLLECTION_DOCUMENTS) else None

    # 2. Przeszukiwanie bazy OBRAZÓW (OpenCLIP)
    img_response = qdrant_client.query_points(
        collection_name=COLLECTION_IMAGES,
        query=query_vector_image,
        limit=limit_per_category,
        score_threshold=img_score_threshold
    ) if qdrant_client.collection_exists(COLLECTION_IMAGES) else None

    # Deduplikacja chunków w dokumentach (max 1 trafienie per plik)
    doc_results = []
    seen_doc_files = set()
    if doc_response:
        for hit in doc_response.points:
            payload = hit.payload or {}
            file_id = payload.get("file_id", "Nieznany plik")
            if file_id in seen_doc_files:
                continue
            seen_doc_files.add(file_id)
            doc_results.append({
                "score": round(hit.score, 4),
                "file_id": file_id,
                "chunk_id": payload.get("chunk_id", 0),
                "ocr_text": payload.get("ocr_text", ""),
                "page_number": payload.get("page_number", 1)
            })
            if len(doc_results) >= limit_per_category:
                break

    # Formatowanie wyników dla czystych obrazów
    img_results = []
    if img_response:
        for hit in img_response.points:
            payload = hit.payload or {}
            img_results.append({
                "score": round(hit.score, 4),
                "file_id": payload.get("file_id", "Nieznany plik")
            })

    return {
        "documents": doc_results,
        "images": img_results
    }


def print_search_results(results: Dict[str, List[Dict[str, Any]]], query: str):
    print("\n" + "=" * 70)
    print(f" WYSZUKIWANIE MULTIMODALNE DLA: '{query}'")
    print("=" * 70)

    # Sekcja Dokumentów Tekstowych
    print("\n DOKUMENTY TEKSTOWE / OCR (Top Trafienia):")
    print("-" * 50)
    if results["documents"]:
        for idx, doc in enumerate(results["documents"], start=1):
            print(f"[{idx}] Plik: {doc['file_id']} | Score: {doc['score']} | Strona: {doc['page_number']}")
            if doc["ocr_text"]:
                short_text = doc["ocr_text"][:110].replace('\n', ' ')
                print(f"    Tekst: \"{short_text}...\"")
    else:
        print("Brak pasujących dokumentów tekstowych.")

    # Sekcja Obrazów Wizualnych
    print("\n CZYSTE OBRAZY I ZDJĘCIA (Top Trafienia):")
    print("-" * 50)
    if results["images"]:
        for idx, img in enumerate(results["images"], start=1):
            print(f"[{idx}] Plik: {img['file_id']} | Score: {img['score']}")
    else:
        print("Brak pasujących obrazów.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    qdrant_path = "./qdrant_db"
    if not Path(qdrant_path).exists():
        logger.error("Najpierw uruchom database_creator.py!")
        exit(1)

    client = QdrantClient(path=qdrant_path)
    embedder = EmbeddingComponent(allow_cloud=False)

    while True:
        try:
            user_query = input("Szukaj > ").strip()
            if not user_query or user_query.lower() in ("exit", "quit", "q"):
                break

            results = search_multimodal(
                query_text=user_query,
                qdrant_client=client,
                embedder=embedder,
                limit_per_category=3
            )
            print_search_results(results, user_query)
        except KeyboardInterrupt:
            break

    client.close()