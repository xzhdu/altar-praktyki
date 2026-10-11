import json
import time
from pathlib import Path
from PIL import Image
import fitz 
import torch
import psutil
from transformers import AutoProcessor, AutoModel

# Configs
MODEL_NAME = "SigLIP 2"
MODEL_ID = "google/siglip2-base-patch16-224"
DATASET_DIR = Path(r"C:\Users\mateu\Desktop\Zbiór_danych")
RESULTS_DIR = Path(r"C:\Users\mateu\Desktop\Embedding_SigLip2\SigLip2_Base_FixRes.json")
QUERIES_DIR = DATASET_DIR / "Queries"
EXCLUDED_FOLDERS = {"document_scans"}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.pdf'}

# Check if CUDA is available and set the device accordingly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model and processor from Hugging Face
def load_model():
    print(f"Ładowanie modelu {MODEL_ID} na urządzenie: {DEVICE}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


# Get all valid file paths from the dataset directory, excluding specified folders and queries
def get_all_file_paths(dataset_dir: Path, excluded_folders: set) -> list[Path]:
    file_paths = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            parts_lower = [p.lower() for p in path.parts]
            if any(ex.lower() in parts_lower for ex in excluded_folders) or "queries" in parts_lower:
                continue
            file_paths.append(path)

    print(f"Znaleziono {len(file_paths)} plików (grafiki + PDF) do bazy danych.")
    return file_paths


# Load image from file path, handling both image and PDF formats
def load_as_pil_image(file_path: Path) -> Image.Image:
    if file_path.suffix.lower() == '.pdf':
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                raise ValueError("Plik PDF jest pusty.")
            page = doc[0]
            pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        with Image.open(file_path) as img:
            return img.convert("RGB")


# Load all queries from JSON files in the specified directory
def load_all_queries_from_dir(queries_dir: Path) -> list[dict]:
    all_queries = []
    if not queries_dir.exists():
        print(f"OSTRZEŻENIE: Folder z zapytaniami '{queries_dir}' nie istnieje!")
        return all_queries

    json_files = list(queries_dir.glob("*.json"))
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                for target_file, query_list in data.items():
                    if isinstance(query_list, list):
                        for q in query_list:
                            all_queries.append({
                                "target_file": target_file,
                                "query": q
                            })
        except Exception as e:
            print(f"Błąd wczytywania {file_path.name}: {e}")

    print(f"Łącznie wczytano {len(all_queries)} zapytań tekstowych do przetestowania.")
    return all_queries


# Generate image embedding using the model and processor
def get_image_embedding(file_path: Path, model, processor):
    try:
        image = load_as_pil_image(file_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)

            if not isinstance(features, torch.Tensor):
                features = getattr(features, "pooler_output", features[0])

            normalized_features = features / features.norm(dim=-1, keepdim=True)

        return normalized_features.squeeze(0)
    except Exception as e:
        print(f"Błąd generowania embeddingu dla {file_path.name}: {e}")
        return None


# Generate text embedding using the model and processor, also measuring execution time
def get_text_embedding_with_time(text: str, model, processor):
    start_time = time.perf_counter()
    
    inputs = processor(text=[text], padding="max_length", max_length=64, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        
        if not isinstance(features, torch.Tensor):
            features = getattr(features, "pooler_output", features[0])

        normalized_features = features / features.norm(dim=-1, keepdim=True)

    exec_time = (time.perf_counter() - start_time) * 1000  # ms
    return normalized_features.squeeze(0), exec_time


# Main function to run the benchmark
def main():
    print("=== ROZPOCZĘCIE BENCHMARKU SIGLIP 2 ===")

    # Load the model and processor
    model, processor = load_model()

    params_count = sum(p.numel() for p in model.parameters()) / 1e6
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # 2. Generate embeddings for all images in the dataset
    file_paths = get_all_file_paths(DATASET_DIR, EXCLUDED_FOLDERS)
    image_names = []
    image_embeddings_list = []

    print("\n[1/2] Tworzenie bazy embeddingów obrazów...")
    for idx, path in enumerate(file_paths, 1):
        print(f"[{idx}/{len(file_paths)}] Generowanie embeddingu: {path.name}...")
        emb = get_image_embedding(path, model, processor)
        if emb is not None:
            image_names.append(path.name)
            image_embeddings_list.append(emb)

    if not image_embeddings_list:
        print("Błąd: Nie udało się wygenerować żadnych embeddingów dla obrazów.")
        return
    
    embedding_dim = image_embeddings_list[0].shape[0]
    print(f"Wymiar wektora (embedding dimension): {embedding_dim}")

    # 3. Stack all image embeddings into a single tensor for efficient similarity computation
    image_matrix = torch.stack(image_embeddings_list)

    queries = load_all_queries_from_dir(QUERIES_DIR)

    # 4. Perform queries and search for the correct image
    benchmark_results = []
    print("\n[2/2] Wykonywanie zapytań i wyszukiwanie...")

    for idx, q_info in enumerate(queries, 1):
        target_file = q_info["target_file"]
        query_text = q_info["query"]

        # 4a. Generate text embedding and measure time
        text_emb, embed_gen_time_ms = get_text_embedding_with_time(query_text, model, processor)

        # 4b. Cosine similarity search between text embedding and all image embeddings
        search_start = time.perf_counter()
        similarities = torch.matmul(image_matrix, text_emb)
        sorted_indices = torch.argsort(similarities, descending=True).cpu().tolist()
        search_time_ms = (time.perf_counter() - search_start) * 1000

        # 4c. Find the rank of the correct image in the sorted results
        correct_position = None
        target_filename = Path(target_file).name.lower()
        for rank, img_idx in enumerate(sorted_indices, 1):
            if Path(image_names[img_idx]).name.lower() == target_filename:
                correct_position = rank
                break

        # 4d. Defining format of the benchmark result for this query
        benchmark_results.append({
            "model_name": MODEL_NAME,
            "image_id": target_file,
            "query": query_text,
            "correct_result_position": correct_position,
            "embedding_generation_time_ms": round(embed_gen_time_ms, 2),
            "search_time_ms": round(search_time_ms, 2)
        })

        print(f"[{idx}/{len(queries)}] Zapytanie: '{query_text[:30]}...' -> Wzorzec ({target_file}) na pozycji: {correct_position}")

    # 5. Save benchmark results to JSON file
    RESULTS_DIR.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    # EXTRACTING ADDITIONAL INFORMATION ABOUT THE MODEL
    process = psutil.Process()
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    print("\n" + "=" * 50)
    print("✅ BENCHMARK ZAKOŃCZONY SUKCESEM!")
    print(f"• Zużycie pamięci RAM: {ram_usage_mb:.2f} MB")
    print(f"• Wymiar wektora (embedding dim): {embedding_dim}")
    print(f"• Rozmiar modelu: {params_count:.2f} M parametrów ({model_size_mb:.2f} MB)")
    print(f"• Wyniki zapisano w: {RESULTS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()