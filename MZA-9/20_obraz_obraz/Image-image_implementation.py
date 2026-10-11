import os
import open_clip
from PIL import Image
from qdrant_client import QdrantClient
import torch

IMG_BASE_DIR = r"C:\Users\mateu\Desktop\Zbiór_danych"
QDRANT_BASE_DIR = r"C:\Users\mateu\Desktop\Baza_wektorowa\qdrant_db"
COLLECTION_NAME = "image_embeddings"

MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"


def load_search_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ładowanie modelu na urządzeniu: {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, device

def get_image_embedding(image_path, model, preprocess, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = model.encode_image(image_tensor)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
    return img_embedding.squeeze(0).cpu().tolist()

def search_in_qdrant(query_vector, top_k=3):
    client = QdrantClient(path=QDRANT_BASE_DIR)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    client.close()
    return response.points


def display_search_results(search_results, img_base_dir):
    if not search_results:
        print("Brak wyników pasujących do zapytania.")
        return

    print("\nWyniki wyszukiwania:")
    top_file_name = None

    for idx, result in enumerate(search_results, start=1):
        payload = result.payload or {}
        file_name = payload.get("file_name", "Nieznany")
        model_used = payload.get(
            "model_used", payload.get("model_name", "Nieznany")
        )
        score_in_percent = result.score * 100

        print(
            f"{idx}. Plik: {file_name} | Dopasowanie: {score_in_percent:.2f}% | Model: {model_used}"
        )

        if idx == 1:
            top_file_name = file_name

    top_result_path = None
    if top_file_name:
        for root, _, files in os.walk(img_base_dir):
            if top_file_name in files:
                top_result_path = os.path.join(root, top_file_name)
                break

    if top_result_path and os.path.exists(top_result_path):
        print(f"Otwieram podgląd: {os.path.basename(top_result_path)}")
        try:
            img = Image.open(top_result_path)
            img.show()
        except Exception as e:
            print(f"Błąd podczas otwierania obrazu: {e}")
    else:
        print("Nie odnaleziono pliku graficznego na dysku.")


def main():
    model, preprocess, device = load_search_model()
    print("Wyszukiwarka uruchomiona. Wpisz zapytanie lub 'q' aby zakończyć.")

    while True:
        user_input = (
            input("\nPodaj ścieżkę do pliku graficznego: ").strip().strip('"').strip("'"))
        if user_input.lower() == "q":
            print("Zakończenie programu.")
            break

        if not user_input:
            continue

        query_vector = get_image_embedding(user_input, model, preprocess, device)
        search_results = search_in_qdrant(query_vector, top_k=3)
        display_search_results(search_results, IMG_BASE_DIR)


if __name__ == "__main__":
    main()