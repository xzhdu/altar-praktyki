import os
from pathlib import Path
import fitz
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import torch
from transformers import AutoModel, AutoProcessor

# Configs
MODEL_ID = "google/siglip2-base-patch16-naflex"
DATASET_DIR = Path(r"C:\Users\mateu\Desktop\Zbiór_danych")
QDRANT_BASE_DIR = r"C:\Users\mateu\Desktop\Baza_wektorowa_siglip2\qdrant_db"
COLLECTION_NAME = "images_embeddings_siglip2"

EXCLUDED_FOLDERS = {"document_scans"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".pdf"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Ładowanie modelu {MODEL_ID} na urządzenie: {DEVICE}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def get_all_file_paths(dataset_dir: Path, excluded_folders: set) -> list[Path]:
    file_paths = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            parts_lower = [p.lower() for p in path.parts]
            if (
                any(ex.lower() in parts_lower for ex in excluded_folders)
                or "queries" in parts_lower
            ):
                continue
            file_paths.append(path)
    return file_paths


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
    try:
        image = load_as_pil_image(file_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = getattr(features, "pooler_output", features[0])
            normalized_features = features / features.norm(
                dim=-1, keepdim=True
            )

        return normalized_features.squeeze(0).cpu().tolist()
    except Exception as e:
        print(f"Błąd generowania embeddingu dla {file_path.name}: {e}")
        return None


def main():
    print("=== TWORZENIE BAZY SIGLIP 2 W QDRANCIE ===")
    model, processor = load_model()
    file_paths = get_all_file_paths(DATASET_DIR, EXCLUDED_FOLDERS)
    print(f"Znaleziono {len(file_paths)} plików do zindeksowania.")

    client = QdrantClient(path=QDRANT_BASE_DIR)

    # Indeksowanie pierwszej grafiki w celu dynamicznego pobrania wymiarowości (np. 768)
    first_emb = None
    for p in file_paths:
        first_emb = get_image_embedding(p, model, processor)
        if first_emb is not None:
            break

    if first_emb is None:
        print("Nie udało się wygenerować żadnego wektora testowego.")
        return

    vector_size = len(first_emb)
    print(f"Wymiarowość wektora SigLIP 2: {vector_size}")

    # Tworzenie osobnej kolekcji
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE
            ),
        )
        print(f"Utworzono kolekcję '{COLLECTION_NAME}' w Qdrancie.")
    else:
        print(f"Kolekcja '{COLLECTION_NAME}' już istnieje. Dopisuję punkty...")

    points = []
    for idx, path in enumerate(file_paths, 1):
        print(f"[{idx}/{len(file_paths)}] Przetwarzanie: {path.name}...")
        emb = get_image_embedding(path, model, processor)
        if emb is None:
            continue

        payload = {
            "file_name": path.name,
            "relative_path": str(path.relative_to(DATASET_DIR)),
            "full_path": str(path),
            "model_name": MODEL_ID,
        }

        points.append(PointStruct(id=idx, vector=emb, payload=payload))

    print(f"\nZapisywanie {len(points)} wektorów do Qdranta...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    client.close()
    print("✅ Zakończono tworzenie bazy SigLIP 2!")


if __name__ == "__main__":
    main()