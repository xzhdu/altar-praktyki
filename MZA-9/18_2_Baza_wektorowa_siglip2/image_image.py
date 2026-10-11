import os
from pathlib import Path
import fitz
from PIL import Image
from qdrant_client import QdrantClient
import torch
from transformers import AutoModel, AutoProcessor

# Konfiguracja
MODEL_ID = "google/siglip2-base-patch16-naflex"
QDRANT_BASE_DIR = r"C:\Users\mateu\Desktop\Baza_wektorowa_siglip2\qdrant_db"
COLLECTION_NAME = "images_embeddings_siglip2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Ładowanie modelu {MODEL_ID} na urządzenie: {DEVICE}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def load_as_pil_image(file_path: Path) -> Image.Image:
    if file_path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                raise ValueError("Plik PDF jest pusty.")
            page = doc[0]
            pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        with Image.open(file_path) as img:
            return img.convert("RGB")


def get_image_embedding(file_path: Path, model, processor):
    image = load_as_pil_image(file_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = getattr(features, "pooler_output", features[0])
        normalized_features = features / features.norm(dim=-1, keepdim=True)

    return normalized_features.squeeze(0).cpu().tolist()


def search_in_qdrant(query_vector, top_k=5):
    client = QdrantClient(path=QDRANT_BASE_DIR)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    client.close()
    return response.points


def display_results(results):
    if not results:
        print("Brak wyników pasujących do wskazanego pliku.")
        return

    print("\n--- WYNIKI WYSZUKIWANIA OBRAZ -> OBRAZ ---")
    top_image_path = None

    for idx, res in enumerate(results, start=1):
        payload = res.payload or {}
        file_name = payload.get("file_name", "Nieznany")
        score_percent = res.score * 100
        full_path = payload.get("full_path")

        print(f"{idx}. {file_name} | Dopasowanie: {score_percent:.2f}%")

        if idx == 1 and full_path and os.path.exists(full_path):
            top_image_path = full_path

    if top_image_path:
        try:
            img = load_as_pil_image(Path(top_image_path))
            img.show()
        except Exception as e:
            print(f"Nie udało się otworzyć podglądu: {e}")


def main():
    model, processor = load_model()
    print("\n=== WYSZUKIWARKA OBRAZ -> OBRAZ (SIGLIP 2) ===")
    print("Podaj ścieżkę do pliku graficznego lub PDF. Aby zakończyć, wpisz 'q'.\n")

    while True:
        user_input = (
            input("\nPodaj ścieżkę do obrazu wejściowego: ")
            .strip()
            .strip('"')
            .strip("'")
        )

        if user_input.lower() == "q":
            print("Zakończenie pracy.")
            break

        if not user_input:
            continue

        input_path = Path(user_input)

        if not input_path.exists() or not input_path.is_file():
            print(f"Błąd: Plik pod wskazaną ścieżką nie istnieje ({user_input}).")
            continue

        try:
            query_vec = get_image_embedding(input_path, model, processor)
            results = search_in_qdrant(query_vec, top_k=5)
            display_results(results)
        except Exception as e:
            print(f"Błąd przetwarzania obrazu: {e}")


if __name__ == "__main__":
    main()