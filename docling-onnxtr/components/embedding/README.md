# OpenCLIP Embedding Component

Lekki, modułowy komponent w języku Python przeznaczony do generowania wektorów (embeddingów) dla tekstu oraz obrazów za pomocą biblioteki OpenCLIP. Kod został zaprojektowany z myślą o pracy w środowiskach z ograniczonymi zasobami sprzętowymi oraz łatwej integracji z bazami wektorowymi.

## Funkcjonalności

* **Multimodalność:** Generowanie spójnych przestrzenie wektorowych dla tekstu i obrazu (`xlm-roberta-base-ViT-B-32`).
* **Format Metadanych:** Metoda `get_metadata()` generuje ustandaryzowany słownik gotowy do zapisu w bazie danych.
* **Obsługa Błędów:** Dedykowane wyjątki (`ImageProcessingError`, `CloudModelDisabledError`) ułatwiające kontrolowanie działania w aplikacjach nadrzędnych.
* **Bezpiecznik Chmurowy:** Domyślnie zablokowane wywołania chmurowe (`allow_cloud=False`), zapobiegające niekontrolowanym kosztom API.
* **Optymalizacja VRAM:** Automatyczna detekcja akceleracji sprzętowej (CUDA/CPU) oraz tryb ewaluacji (`eval()`) bez obliczania gradientów.

## Struktura Plików

* `component.py` – Główny moduł zawierający klasę `EmbeddingComponent` oraz definicje wyjątków.
* `test.py` – Skrypt weryfikacyjny testujący podstawowe przepływy pracy, generowanie metadanych oraz obsługę błędów, pokazujący użycie najważniejszych metod komponentu

## Najważniejsze metody

* `embed_image()` - Służy do przekształcania obrazów na wektor
* `embed_text()` - Służy do przekształcania tekstu na wektor
* `get_metadata()` - służy do pozyskiwania metadanych wektora

## Wymagania i Instalacja

Do uruchomienia modułu wymagany jest Python oraz następujące pakiety:

```bash
pip install torch open_clip_torch pillow
