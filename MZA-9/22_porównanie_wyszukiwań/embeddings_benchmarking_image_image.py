import os
import platform
import subprocess
import torch
import open_clip
from PIL import Image
from qdrant_client import QdrantClient

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"

print(f"Ładowanie modelu {model_name} na urządzeniu: {device}...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
model = model.to(device)
model.eval()

# 2. Połączenie z bazą Qdrant
client = QdrantClient(path="./qdrant_db")

COLLECTIONS = {
    "1. Wizualny": "variant_1_visual",
    "2. Opis (Mistral)": "variant_2_text_desc",
    "3. Hybrydowy": "variant_3_hybrid",
}


def encode_image_query(img_path: str) -> list[float]:
  image = Image.open(img_path).convert("RGB")
  tensor = preprocess(image).unsqueeze(0).to(device)
  with torch.no_grad():
    feats = model.encode_image(tensor)
    feats /= feats.norm(dim=-1, keepdim=True)
  return feats.squeeze(0).cpu().numpy().tolist()


def open_image(file_path: str):
  if not file_path or not os.path.exists(file_path):
    print(
        f"\n[BŁĄD] Nie można otworzyć pliku (ścieżka nie istnieje): {file_path}"
    )
    return

  sys_name = platform.system()
  try:
    print(f"\n Otwieranie pliku wynikowego: {file_path} ...")
    if sys_name == "Windows":
      os.startfile(file_path)
    elif sys_name == "Darwin":  # macOS
      subprocess.run(["open", file_path])
    else:  # Linux
      subprocess.run(["xdg-open", file_path])
  except Exception as e:
    print(f"[BŁĄD przy otwieraniu obrazu]: {e}")


def main():
  print("\n" + "=" * 70)
  print(" Podaj ścieżkę do obrazu wzorcowego i naciśnij Enter.")
  print(" Wpisz 'q', 'exit' lub 'quit', aby zakończyć działanie.")
  print("=" * 70)

  TOP_K = 3

  while True:
    try:
      img_input = (
          input("\nPodaj ścieżkę do obrazu zapytania: ").strip().strip('"\'')
      )
    except (KeyboardInterrupt, EOFError):
      print("\nZakończono działanie.")
      break

    if not img_input:
      continue

    if img_input.lower() in ["q", "exit", "quit"]:
      print("Zakończono działanie programu.")
      break

    if not os.path.exists(img_input):
      print(
          f"[BŁĄD] Plik o ścieżce '{img_input}' nie istnieje. Spróbuj ponownie."
      )
      continue

    try:
      query_vector = encode_image_query(img_input)
    except Exception as e:
      print(f"[BŁĄD przy przetwarzaniu obrazu]: {e}")
      continue

    print("\n" + "=" * 80)
    print(f" WYNIKI WYSZUKIWARKI DLA OBRAZU: '{img_input}'")
    print("=" * 80)

    best_overall_score = -1.0
    best_overall_file_path = None
    best_overall_variant = ""

    for var_name, coll_name in COLLECTIONS.items():
      print(f"\n--- Wariant: {var_name} ---")

      res = client.query_points(
          collection_name=coll_name, query=query_vector, limit=TOP_K
      )

      if not res.points:
        print("  Brak wyników w bazie.")
        continue

      for rank, point in enumerate(res.points, 1):
        file_name = point.payload.get("file_name", "Brak nazwy")
        full_path = point.payload.get("full_path", "")
        img_type = point.payload.get("type", "nieznany")
        score = point.score * 100

        print(
            f"  {rank}. Plik: {file_name:<18} | Kategoria: {img_type:<12} |"
            f" Dopasowanie: {score:.2f}%"
        )

        if (
            score > best_overall_score
            and os.path.abspath(full_path) != os.path.abspath(img_input)
        ):
          best_overall_score = score
          best_overall_file_path = full_path
          best_overall_variant = var_name

    if best_overall_file_path:
      print("\n" + "-" * 80)
      print(" NAJLEPSZE PODOBNE ZDJĘCIE Z BAZY:")
      print(
          f" Wariant: {best_overall_variant} | Score:"
          f" {best_overall_score:.2f}%"
      )
      open_image(best_overall_file_path)
    else:
      print("\n[INFO] Brak innych pasujących zdjęć w bazie.")


if __name__ == "__main__":
  main()