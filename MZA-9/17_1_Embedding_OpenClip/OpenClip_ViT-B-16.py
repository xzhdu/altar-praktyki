import os
import glob
import json
import time
import psutil  # RAM
import torch
import open_clip
from PIL import Image
import pypdfium2 as pdfium

# 1. Konfiguracja ścieżek
DATASET_DIR =r"C:\Users\mateu\Desktop\Zbiór_danych"  # Pobrana lokalnie i zmodyfikowana ścieżka do folderu z danymi, dopasowana do mojego zadania
QUERIES_DIR = os.path.join(DATASET_DIR, "Queries")
CUSTOM_OUTPUT_PATH = r"C:\Users\mateu\Desktop\Zbiór_danych\Wyniki_ewaluacji_OpenCLIP16.json"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"


def load_image_or_pdf(file_path):
    """Wczytuje plik JPG/PNG lub renderuje pierwszą stronę pliku PDF i natychmiast zamyka plik."""
    if file_path.lower().endswith('.pdf'):
        pdf = pdfium.PdfDocument(file_path)
        try:
            page = pdf[0]
            # Renderujemy do obrazu PIL i robimy kopie w pamięci (.copy()), 
            # aby bezpiecznie zamknąć plik PDF z dysku
            pil_image = page.render(scale=2).to_pil().convert("RGB").copy()
        finally:
            pdf.close()  # <- Niwelujemyy komunikat o automatycznym zamykaniu pliku PDF
        return pil_image
    else:
        with Image.open(file_path) as img:
            return img.convert("RGB").copy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu" #wybieramy optymalne urządzenie do obliczeń (GPU jeśli dostępne, w przeciwnym razie CPU)
    print(f" Uruchamianie na urządzeniu: {device}")

    # Ładowanie modelu OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    # Statystyki modelu i środowiska, które były wymagane w poleceniu
    # 1. Wielkość modelu
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    # 2. Zużycie pamięci RAM
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    # 3. Wymiar embeddingu
    embedding_dim = model.text_projection.shape[1] if hasattr(model, 'text_projection') else 512

    print("\n" + "=" * 40)
    print("DANE DO RAPORTU:")
    print(f" - Wymiar embeddingu: {embedding_dim}")
    print(f" - Wielkość modelu w pamięci: {model_size_mb:.2f} MB")
    print(f" - Zużycie pamięci RAM programu: {ram_mb:.2f} MB")
    print("=" * 40 + "\n")
    # ----------------------------------------

    # 2. Mapowanie plików z podfolderów
    image_paths_map = {}
    subfolders = ["app_screenshots", "charts", "pdf_files", "photos"]

    for folder in subfolders:
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.exists(folder_path):
            for file_path in glob.glob(os.path.join(folder_path, "*.*")):
                filename = os.path.basename(file_path)
                image_paths_map[filename] = file_path

    print(f"Znaleziono {len(image_paths_map)} plików w folderach ze zdjęciami/PDF.")

    # 3. Indeksowanie bazy obrazów (Generowanie embeddingów dla WSZYSTKICH plików)
    image_ids = []
    image_tensors_list = []

    print("Tworzenie bazy wektorowej obrazów...")
    for img_id, full_path in image_paths_map.items():
        try:
            pil_img = load_image_or_pdf(full_path)
            processed_img = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_emb = model.encode_image(processed_img)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

            image_ids.append(img_id)
            image_tensors_list.append(img_emb)
        except Exception as e:
            print(f"Błąd podczas przetwarzania {img_id}: {e}")

    all_images_matrix = torch.cat(image_tensors_list, dim=0)

    # 4. Odczytywanie wszystkich zapytań z folderu Queries
    query_files = glob.glob(os.path.join(QUERIES_DIR, "*.json"))
    results = []

    print("Rozpoczynanie testów zapytań...")
    for q_file in query_files:
        with open(q_file, 'r', encoding='utf-8') as f:
            try:
                queries_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Błąd składni JSON w pliku {q_file}: {e}")
                continue

        for target_image_id, queries in queries_data.items():
            if target_image_id not in image_ids:
                print(f"Ostrzeżenie: Brak pliku {target_image_id} w bazie plików graficznych!")
                continue

            correct_index = image_ids.index(target_image_id)

            for q_text in queries:
                # --- Pomiar czasu generowania embeddingu tekstu ---
                t0_emb = time.perf_counter()
                tokens = tokenizer([q_text]).to(device)
                with torch.no_grad():
                    text_emb = model.encode_text(tokens)
                    text_emb /= text_emb.norm(dim=-1, keepdim=True)
                t1_emb = (time.perf_counter() - t0_emb) * 1000

                # --- Pomiar czasu wyszukiwania ---
                t0_search = time.perf_counter()
                similarity_scores = (text_emb @ all_images_matrix.T).squeeze(0)
                sorted_indices = torch.argsort(similarity_scores, descending=True).tolist()
                t1_search = (time.perf_counter() - t0_search) * 1000

                rank = sorted_indices.index(correct_index) + 1

                results.append({
                    "model_name": f"OpenCLIP {MODEL_NAME}",
                    "image_id": target_image_id,
                    "query": q_text,
                    "correct_result_position": rank,
                    "embedding_generation_time_ms": round(t1_emb, 2),
                    "search_time_ms": round(t1_search, 2)
                })

    # 5. Zapis wyników do pliku JSON
    CUSTOM_OUTPUT_PATH = r"C:\Users\mateu\Desktop\Zbiór_danych\Wyniki_ewaluacji_OpenCLIP16.json"

    with open(CUSTOM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSukces! Wyniki zapisano do: {CUSTOM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()