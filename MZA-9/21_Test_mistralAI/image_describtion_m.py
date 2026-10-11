import base64
import json
import mimetypes
import os
import time
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel, Field

# 1. Wczytanie klucza API z pliku .env
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("Nie odnaleziono klucza MISTRAL_API_KEY w pliku .env!")

client = Mistral(api_key=api_key)


# 2. Definicja typów i modeli Pydantic
class ImageType(str, Enum):
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    CHART = "chart"
    DIAGRAM = "diagram"


class ImageAnalysisResult(BaseModel):
    type: ImageType
    summary: str
    visibleText: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


# 3. Funkcja pomocnicza do kodowania obrazu w base64
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# 4. Funkcja do analizy pojedynczego obrazu przez API Mistral (składnia MistralClient)
def analyze_image(image_path: Path, max_retries: int = 3) -> ImageAnalysisResult | None:
    base64_image = encode_image(image_path)

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    prompt = """Przeanalizuj szczegółowo podany obraz i wygeneruj odpowiedź w formacie JSON zgodną ze schematem:
{
  "type": "photo" | "screenshot" | "chart" | "diagram",
  "summary": "Krótki, ale szczegółowy opis zawartości po polsku",
  "visibleText": ["odczytany tekst 1", "odczytany tekst 2"],
  "keywords": ["słowo kluczowe 1", "słowo kluczowe 2"]
}

Zasady:
1. "type" musi mieć dokładnie jedną z wartości: "photo", "screenshot", "chart", "diagram".
2. Jeśli na obrazie nie ma czytelnego tekstu, "visibleText" ma być pustą listą [].
3. Wynik musi być czystym obiektem JSON."""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.complete(
                model="pixtral-12b-2409",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime_type};base64,{base64_image}",
                            },
                        ],
                    }
                ],
            )

            raw_json = response.choices[0].message.content

            if not raw_json:
                raise ValueError("Otrzymano pustą odpowiedź z API Mistral.")

            return ImageAnalysisResult.model_validate_json(raw_json)
            raw_json = response.choices[0].message.content

            if not raw_json:
                raise ValueError("Otrzymano pustą odpowiedź z API Mistral.")

            return ImageAnalysisResult.model_validate_json(raw_json)

        except Exception as e:
            print(
                f" [Próba {attempt}/{max_retries}] Błąd przy pliku {image_path.name}: {e}"
            )
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(
                    f" Ostatecznie pominięto plik {image_path.name} po {max_retries} nieudanych próbach."
                )
                return None


# 5. Główna funkcja przetwarzająca foldery i zapisująca wyniki
def process_selected_folders(
    base_dir: str, target_subfolders: list[str], output_json: str):
    base_path = Path(base_dir)
    output_path = Path(output_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = []

    for subfolder in target_subfolders:
        folder_path = base_path / subfolder
        if folder_path.exists() and folder_path.is_dir():
            files = [
                p
                for p in folder_path.rglob("*")
                if p.suffix.lower() in valid_extensions
            ]
            image_files.extend(files)
        else:
            print(f"Ostrzeżenie: Podfolder '{subfolder}' nie istnieje!")

    print(
        f"Znaleziono łącznie {len(image_files)} obrazów w podfolderach: {', '.join(target_subfolders)}"
    )

    results = []
    processed_paths = set()

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_paths = {item["full_path"] for item in results}
            print(
                f"Wczytano istniejący plik wyników. Pomijam {len(processed_paths)} już przetworzonych plików."
            )
        except Exception as e:
            print(
                f"Nie udało się odczytać istniejącego pliku JSON ({e}). Zaczynam od zera."
            )

    for idx, img_path in enumerate(image_files, start=1):
        full_path_str = str(img_path.resolve())

        if full_path_str in processed_paths:
            print(
                f"[{idx}/{len(image_files)}] Pominięto (już przetworzono): {img_path.name}"
            )
            continue

        print(f"\n[{idx}/{len(image_files)}] Przetwarzanie: {img_path.name}...")

        analysis_result = analyze_image(img_path)

        if analysis_result:
            record = {
                "file_name": img_path.name,
                "full_path": full_path_str,
                "analysis": analysis_result.model_dump(),
            }
            results.append(record)
            processed_paths.add(full_path_str)

            print(
                f"  -> Typ: {analysis_result.type.value} | Opis: {analysis_result.summary}"
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            print(f"  -> Pominięto plik z powodu błędu.")

    print(
        f"\nGotowe! Łącznie w pliku jest {len(results)} opisów. Zapisano w: {output_json}"
    )


# === PUNKT WEJŚCIA SKRYPTU ===
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\mateu\Desktop\Zbiór_danych"
    TARGET_SUBFOLDERS = ["app_screenshots", "charts", "photos"]
    OUTPUT_FILE = r"C:\Users\mateu\Desktop\Test_mistralAI\descriptions.json"

    process_selected_folders(BASE_DIR, TARGET_SUBFOLDERS, OUTPUT_FILE)