import logging
import time
from PIL import Image

from component import (
    EmbeddingComponent,
    EmbeddingError,
    ImageProcessingError,
    CloudModelDisabledError
)

# 1. Konfiguracja logowania, aby komunikaty z komponentu wyświetlały się w konsoli
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("TestScript")

# 2. Inicjalizacja komponentu (z zablokowaną chmurą allow_cloud=False)
logger.info("Tworzenie instancji EmbeddingComponent...")
embedder = EmbeddingComponent(allow_cloud=False)

# 3. Test embeddingu tekstu
print("\n--- TEST 1: Embedding Tekstu ---")
text_sample = "Wyszukiwanie obrazów w oparciu o tekst jest fascynujące!"

start_time = time.time()
text_vector = embedder.embed_text(text_sample)
exec_time_ms = (time.time() - start_time) * 1000

text_meta = embedder.get_metadata(execution_time_ms=exec_time_ms)

logger.info(f"Tekst: '{text_sample}'")
logger.info(f"Długość wektora: {len(text_vector)}")
logger.info(f"Próbka wektora (pierwsze 3 liczby): {text_vector[:3]}")
logger.info(f"Metadane: {text_meta}")

# 4. Test embeddingu obrazu
print("\n--- TEST 2: Embedding Obrazu z pliku ---")
image_path = "C:\\Users\\mateu\\Desktop\\Zbiór_danych\\photos\\animal_3.jpg"

start_time = time.time()
img_vector = embedder.embed_image(image_path)
exec_time_ms = (time.time() - start_time) * 1000

img_meta = embedder.get_metadata(
    file_path=image_path, 
    execution_time_ms=exec_time_ms
)

logger.info(f"Wygenerowano embedding dla pliku: {image_path}")
logger.info(f"Długość wektora: {len(img_vector)}")
logger.info(f"Metadane: {img_meta}")

# 5. TEST 3: Obsługa błędów (nieistniejący plik)
print("\n--- TEST 3: Obsługa błędów (Brak pliku) ---")
try:
    embedder.embed_image("fajny_plik_123.png")
except ImageProcessingError as e:
    logger.info(f"Pomyślnie przechwycono oczekiwany błąd: {e}")

# 6. TEST 4: Bezpiecznik chmurowy (allow_cloud=False)
print("\n--- TEST 4: Bezpiecznik chmurowy ---")
try:
    embedder.embed_via_cloud_api("Sprawdzam połączenie chmurowe")
except CloudModelDisabledError as e:
    logger.info(f"Pomyślnie przechwycono blokadę chmury: {e}")

print("\n=== Wszystkie testy zakończone sukcesem! ===")