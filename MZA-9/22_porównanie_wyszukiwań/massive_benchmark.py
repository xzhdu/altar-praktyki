import glob
import json
import os
import time
import open_clip
import torch
from qdrant_client import QdrantClient

DATASET_DIR = r"C:\Users\mateu\Desktop\Zbior_danych"
QUERIES_DIR = r"C:\Users\mateu\Desktop\Queries"
CUSTOM_OUTPUT_PATH = r"22_porównanie_wyszukiwań\Full_benchmark\Wyniki_ewaluacji_Qdrant_Wariant3.json"

MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"

QDRANT_PATH = "./qdrant_db" 
COLLECTION_NAME = "variant_3_hybrid" 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Uruchamianie na urządzeniu: {device}")

    print(f"Ładowanie modelu {MODEL_NAME}...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    print(f"Łączenie z bazą Qdrant ({QDRANT_PATH})...")
    client = QdrantClient(path=QDRANT_PATH)

    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
        print(f"Pomyślnie połączono. Liczba punktów w kolekcji '{COLLECTION_NAME}': {total_points}")
    except Exception as e:
        print(f"[BŁĄD] Nie można uzyskać dostępu do kolekcji '{COLLECTION_NAME}': {e}")
        return

    query_files = glob.glob(os.path.join(QUERIES_DIR, "*.json"))
    results = []
    total_queries_processed = 0

    print("\nRozpoczynanie testów zapytań...")
    for q_file in query_files:
        filename = os.path.basename(q_file)
        
        if "pdf" in filename.lower():
            print(f" -> Pomijanie pliku: {filename}")
            continue

        print(f" -> Przetwarzanie pliku: {filename}")
        
        with open(q_file, "r", encoding="utf-8") as f:
            try:
                queries_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"   [BŁĄD] Błąd składni JSON w pliku {filename}: {e}")
                continue

        for target_image_id, queries in queries_data.items():
            for q_text in queries:
             
                t0_emb = time.perf_counter()
                tokens = tokenizer([q_text]).to(device)
                with torch.no_grad():
                    text_emb = model.encode_text(tokens)
                    text_emb /= text_emb.norm(dim=-1, keepdim=True)

                    query_vector = text_emb.squeeze(0).cpu().numpy().tolist()
                t1_emb = (time.perf_counter() - t0_emb) * 1000

                t0_search = time.perf_counter()
                
                res = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=max(total_points, 100) 
                )
                t1_search = (time.perf_counter() - t0_search) * 1000

                rank = -1
                if res.points:
                    for index, point in enumerate(res.points):
                        hit_filename = point.payload.get("file_name")
                        if hit_filename == target_image_id:
                            rank = index + 1
                            break

                results.append({
                    "model_name": f"OpenCLIP {MODEL_NAME} (Qdrant - Wariant 2)",
                    "image_id": target_image_id,
                    "query": q_text,
                    "correct_result_position": rank,
                    "embedding_generation_time_ms": round(t1_emb, 2),
                    "search_time_ms": round(t1_search, 2),
                })
                total_queries_processed += 1

    with open(CUSTOM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"SUKCES! Przeanalizowano {total_queries_processed} zapytań (z 3 plików).")
    print(f"Wyniki ewaluacji zapisano do:\n{CUSTOM_OUTPUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()