# OpenCLIP Embedding Component

Lekki, modułowy komponent w języku Python przeznaczony do generowania wektorów (embeddingów) dla tekstu oraz obrazów za pomocą biblioteki OpenCLIP. Kod został zaprojektowany z myślą o pracy w środowiskach z ograniczonymi zasobami sprzętowymi oraz łatwej integracji z bazami wektorowymi.

## Funkcjonalności

* **Multimodalność:** Generowanie spójnej przestrzeni wektorowej dla tekstu i obrazu.
* **Format Metadanych:** Metoda `get_metadata()` generuje ustandaryzowany słownik gotowy do zapisu w bazie danych.
* **Obsługa Błędów:** Dedykowane wyjątki (`EmbeddingError`, `ImageProcessingError`, `CloudModelDisabledError`) ułatwiające kontrolowanie działania w aplikacjach nadrzędnych.
* **Bezpiecznik Chmurowy:** Domyślnie zablokowane wywołania chmurowe (`allow_cloud=False`), zapewniające pełną prywatność i lokalne przetwarzanie danych.
* **Optymalizacja VRAM:** Automatyczna detekcja akceleracji sprzętowej (CUDA/CPU) oraz tryb ewaluacji (`eval()`) bez obliczania gradientów.

## Wybór Domyślnego Modelu

Domyślnie komponent wykorzystuje model **`xlm-roberta-base-ViT-B-32`** z wagami **`laion5b_s13b_b90k`**.

* **Obsługa języka polskiego:** Dzięki tekstowemu enkoderowi XLM-RoBERTa, model bardzo dobrze radzi sobie z zapytaniami w języku polskim oraz angielskim, znacząco przewyższając w analizach modele z rodziny SigLip.
* **Lokalna wydajność:** Generuje zwarte 512-wymiarowe wektory, co zapewnia niski czas wnioskowania (inferencji) na CPU/GPU oraz mniejsze zużycie pamięci w bazie wektorowej.
* **Link do modelu:** [laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k w serwisie Hugging Face](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k)

## Struktura Plików

* `component.py` – Główny moduł zawierający klasę `EmbeddingComponent` oraz definicje dedykowanych wyjątków.
* `test.py` – Skrypt weryfikacyjny testujący podstawowe przepływy pracy, generowanie metadanych oraz obsługę błędów na przykładowych plikach.
* `samples/` – Katalog z przykładowymi obrazami wykorzystywanymi do testów lokalnych.
* `pyproject.toml` / `uv.lock` / `.python-version` – Pliki konfiguracyjne środowiska i zależności.

## Najważniejsze Metody

* `embed_image(image_input)` – Przekształca obraz (ścieżka do pliku lub obiekt `PIL.Image`) na wektor numeryczny.
* `embed_text(text)` – Przekształca ciąg tekstowy na wektor numeryczny.
* `get_metadata(file_path, execution_time_ms)` – Generuje spójny słownik metadanych gotowy do dołączenia do bazy wektorowej.

## Zarządzanie środowiskiem i zależnościami (`uv`)

Projekt wykorzystuje menedżer [uv](https://docs.astral.sh/uv/) do deterministycznego i szybkiego zarządzania zależnościami oraz wersją Pythona. Za pomocą uv sync można zsynchronizować środowiska, dla pełnej zgodności i stabilnego działania

### Wymagania wstępne

Zainstaluj narzędzie `uv`, jeśli nie jest jeszcze dostępne w systemie:
```bash
pip install uv