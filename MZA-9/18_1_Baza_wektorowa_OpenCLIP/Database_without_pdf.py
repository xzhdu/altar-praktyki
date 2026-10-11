import glob
import os
from PIL import Image
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import torch

# 1. Konfiguracja ścieżek
DATASET_DIR = r"C:\Users\mateu\Desktop\Zbiór_danych"
BASE_DIR = r"C:\Users\mateu\Desktop\Baza_wektorowa"
QDRANT_DB_PATH = os.path.join(BASE_DIR, "qdrant_db_wo_pdf")

MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"
REPRESENTATION_TYPE = "FixRes"

ALLOWED_SUBFOLDERS = ["app_screenshots", "charts", "photos"]


def load_image(file_path):
    """Wczytuje plik obrazu (JPG, PNG, WEBP itp.) i zwraca go w formacie RGB."""
    with Image.open(file_path) as img:
        return img.convert("RGB").copy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uruchamianie na urządzeniu: {device}")

    # Ładowanie modelu OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()

    # Ustalamy wymiar wektora (embeddingu) dla danego modelu
    embedding_dim = (
        model.text_projection.shape[1]
        if hasattr(model, "text_projection")
        else 512
    )

    # 2. Inicjalizacja bazy Qdrant
    qdrant = QdrantClient(path=QDRANT_DB_PATH)
    COLLECTION_NAME = "image_embeddings"

    # Tworzenie kolekcji, jeśli jeszcze nie istnieje
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim, distance=Distance.COSINE
            ),
        )
        print(f"Utworzono nową kolekcję Qdrant: {COLLECTION_NAME}")

    # 3. Mapowanie plików z dozwolonych folderów
    image_paths_map = {}
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for folder in ALLOWED_SUBFOLDERS:
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.exists(folder_path):
            for file_path in glob.glob(os.path.join(folder_path, "*.*")):
                if os.path.splitext(file_path)[1].lower() in valid_extensions:
                    filename = os.path.basename(file_path)
                    image_paths_map[filename] = file_path

    print(
        f"Znaleziono {len(image_paths_map)} obrazów do zindeksowania "
        f"w folderach: {', '.join(ALLOWED_SUBFOLDERS)}."
    )

    # 4. Generowanie embeddingów i tworzenie punktów dla Qdrant
    qdrant_points = []
    print("Przetwarzanie obrazów i generowanie embeddingów...")

    for idx, (filename, full_path) in enumerate(
        image_paths_map.items(), start=1
    ):
        try:
            # Wczytanie i preprocessing obrazu
            pil_img = load_image(full_path)
            processed_img = preprocess(pil_img).unsqueeze(0).to(device)

            # Generowanie i normalizacja embeddingu
            with torch.no_grad():
                img_emb = model.encode_image(processed_img)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

            # Konwersja tensora na zwykłą listę wartości float
            vector_list = img_emb.squeeze(0).cpu().tolist()

            # Utworzenie punktu Qdrant zawierającego wymagane dane
            point = PointStruct(
                id=idx,  # Unikalny identyfikator numeryczny
                vector=vector_list,  # Embedding
                payload={
                    "file_name": filename,  # Nazwa pliku
                    "model_name": f"OpenCLIP {MODEL_NAME}",  # Nazwa modelu
                    "representation_type": REPRESENTATION_TYPE,  # Typ reprezentacji (FixRes)
                },
            )
            qdrant_points.append(point)

        except Exception as e:
            print(f"Błąd podczas przetwarzania {filename}: {e}")

    # 5. Zapis (upsert) punktów do bazy wektorowej Qdrant
    if qdrant_points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)
        print(
            f"\nSukces! Zindeksowano {len(qdrant_points)} obrazów."
        )
        print(f"Baza danych została zapisana w: {QDRANT_DB_PATH}")

    qdrant.close()


if __name__ == "__main__":
    main()