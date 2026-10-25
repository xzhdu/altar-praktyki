import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image

from qdrant_client import QdrantClient
from component import EmbeddingComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SearchEngine")

COLLECTION_DOCUMENTS = "documents_ocr"
COLLECTION_IMAGES = "images_visual"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def normalize_image_path(image_path: Union[str, Path]) -> Path:
    path_text = str(image_path).strip()
    if len(path_text) >= 2 and path_text[0] == path_text[-1] and path_text[0] in ("'", '"'):
        path_text = path_text[1:-1].strip()
    return Path(path_text).expanduser()


def search_multimodal(
    query: Union[str, Path, Image.Image],
    qdrant_client: QdrantClient,
    embedder: EmbeddingComponent,
    limit_per_category: int = 3,
    doc_score_threshold: float = None,
    img_score_threshold: float = 0.15
) -> Dict[str, List[Dict[str, Any]]]:

    # 1. Sprawdzenie czy zapytanie to obraz (ścieżka do pliku graficznego lub obiekt PIL.Image)
    is_image_query = False
    image_input = None

    if isinstance(query, Image.Image):
        is_image_query = True
        image_input = query
    elif isinstance(query, (str, Path)):
        path_obj = normalize_image_path(query)
        original_query = str(query)
        looks_like_path = (
            isinstance(query, Path)
            or path_obj.exists()
            or "\\" in original_query
            or "/" in original_query
        )
        if path_obj.suffix.lower() in IMAGE_EXTENSIONS and looks_like_path:
            if not path_obj.is_file():
                raise FileNotFoundError(f"Nie znaleziono pliku obrazu: {path_obj}")
            is_image_query = True
            image_input = path_obj
        elif isinstance(query, Path):
            raise ValueError(
                f"Nieobsługiwany format obrazu '{path_obj.suffix}'. "
                f"Obsługiwane formaty: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )

    # 2. Generowanie odpowiednich wektorów w zależności od typu wejścia
    if is_image_query:
        logger.info(f"Generowanie wektora zapytania dla obrazu: {image_input}")
        # OpenCLIP współdzieli wspólną przestrzeń dla tekstów i obrazów
        visual_vector = embedder.embed_image(image_input)
        query_vector_text = visual_vector
        query_vector_image = visual_vector
    else:
        query_text = str(query)
        logger.info(f"Generowanie wektora zapytania dla tekstu: '{query_text}'")
        query_vector_text = embedder.embed_text(query_text)
        query_vector_image = embedder.embed_text(f"zdjęcie przedstawiające {query_text}")

    # 3. Przeszukiwanie bazy DOKUMENTÓW (OCR)
    doc_response = qdrant_client.query_points(
        collection_name=COLLECTION_DOCUMENTS,
        query=query_vector_text,
        limit=limit_per_category * 3,
        score_threshold=doc_score_threshold
    ) if qdrant_client.collection_exists(COLLECTION_DOCUMENTS) else None

    # 4. Przeszukiwanie bazy OBRAZÓW (OpenCLIP)
    img_response = (
        qdrant_client.query_points(
            collection_name=COLLECTION_IMAGES,
            query=query_vector_image,
            limit=limit_per_category,
            score_threshold=img_score_threshold
        )
        if qdrant_client.collection_exists(COLLECTION_IMAGES)
        else None
    )

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
    qdrant_path = Path(__file__).resolve().parents[1] / "qdrant_db"
    if not qdrant_path.is_dir():
        raise FileNotFoundError(f"Nie znaleziono lokalnej bazy Qdrant: {qdrant_path}")

    client = QdrantClient(path=qdrant_path)
    required_collections = {COLLECTION_DOCUMENTS, COLLECTION_IMAGES}
    available_collections = {
        collection.name for collection in client.get_collections().collections
    }
    missing_collections = required_collections - available_collections
    if missing_collections:
        client.close()
        raise RuntimeError(
            "W bazie Qdrant brakuje kolekcji: "
            f"{', '.join(sorted(missing_collections))}. "
            f"Dostępne kolekcje: {', '.join(sorted(available_collections)) or '(brak)'}"
        )

    embedder = EmbeddingComponent(allow_cloud=False)

    print("=" * 50)
    print(" URUCHOMIONO WYSZUKIWARKĘ MULTIMODALNĄ")
    print("=" * 50)

    while True:
        try:
            print("\nWybierz tryb wyszukiwania:")
            print("  [t] Tekst (wyszukiwanie po frazie)")
            print("  [o] Obraz (wyszukiwanie po pliku graficznym)")
            print("  [q] Wyjście")
            
            mode = input("Wybierz opcję (t / o / q) > ").strip().lower()

            if mode in ("exit", "quit", "q"):
                break

            if mode == "o":
                img_path_str = input("Podaj ścieżkę do pliku obrazu > ").strip()
                if not img_path_str:
                    continue

                query_arg = normalize_image_path(img_path_str)
                if query_arg.suffix.lower() not in IMAGE_EXTENSIONS:
                    print(
                        " Błąd: Nieobsługiwany format obrazu. "
                        f"Dozwolone formaty: {', '.join(sorted(IMAGE_EXTENSIONS))}"
                    )
                    continue
                if not query_arg.is_file():
                    print(f" Błąd: Plik '{query_arg}' nie istnieje lub nie jest plikiem.")
                    continue
            else:
                text_query = input("Wpisz zapytanie tekstowe > ").strip()
                if not text_query:
                    continue
                query_arg = text_query

            results = search_multimodal(
                query=query_arg,
                qdrant_client=client,
                embedder=embedder,
                limit_per_category=3
            )
            print_search_results(results, str(query_arg))
            
        except KeyboardInterrupt:
            print("\nPrzerwano działanie programu.")
            break

    client.close()