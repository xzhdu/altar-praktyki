import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import open_clip
from PIL import Image

# Konfiguracja rejestratora logów
logger = logging.getLogger("EmbeddingComponent")

class EmbeddingError(Exception):
    """Bazowa klasa błędu dla naszego komponentu."""
    pass


class ImageProcessingError(EmbeddingError):
    """Wywołany, gdy plik graficzny jest uszkodzony lub nie istnieje."""
    pass


class CloudModelDisabledError(EmbeddingError):
    """Wywołany, gdy ktoś spróbuje wywołać zablokowany model chmurowy."""
    pass
class EmbeddingComponent:


    def __init__(
        self,
        model_name: str = "xlm-roberta-base-ViT-B-32",
        pretrained: str = "laion5b_s13b_b90k",
        representation_type: str = "FixRes",
        device: Optional[str] = None,
        allow_cloud: bool = False
    ):
        # 1. Zapisywanie konfiguracji w atrybutach obiektu
        self.model_name = model_name
        self.pretrained = pretrained
        self.allow_cloud = allow_cloud
        self.representation_type = representation_type

        # 2. Automatyczny wybór sprzętu (GPU vs CPU)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Inicjalizacja | Model: {self.model_name} | Urządzenie: {self.device} | Cloud: {self.allow_cloud}"
        )

        # 3. Ładowanie modelu z obsługą błędów
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()  # Przełączenie w tryb ewaluacji (inferencji)

            if hasattr(self.model, "text_projection"):
                self.embedding_dim = self.model.text_projection.shape[1]
            else:
                self.embedding_dim = 512

            logger.info(f"Model załadowany. Wymiar embeddingu: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Nie udało się załadować modelu: {e}")
            raise EmbeddingError(f"Błąd inicjalizacji modelu: {e}") from e
    def embed_text(self, text: str) -> List[float]:

         # 1. Walidacja danych wejściowych
        if not text or not text.strip():
            logger.error("Próba przetworzenia pustego tekstu.")
            raise ValueError("Tekst do wektoryzacji nie może być pusty.")

        try:
            # 2. Tokenizacja tekstu i przeniesienie na urządzenie (GPU/CPU)
            tokens = self.tokenizer([text]).to(self.device)

             # 3. Wnioskowanie (inferencja) bez obliczania gradientów
            with torch.no_grad():
                text_emb = self.model.encode_text(tokens)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)

            logger.debug(f"Wygenerowano embedding tekstu dla: '{text[:20]}...'")

            # 4. Konwersja tensora na standardową listę Pythona
            return text_emb.squeeze(0).cpu().tolist()

        except Exception as e:
            logger.error(f"Błąd podczas generowania embeddingu tekstu: {e}")
            raise EmbeddingError(f"Błąd przetwarzania tekstu: {e}") from e
    def embed_image(self, image_input: Union[str, Path, Image.Image]) -> List[float]:

        try:
            # 1. Obsługa różnych typów wejściowych (ścieżka vs plik w pamięci)
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    raise FileNotFoundError(f"Plik obrazu nie istnieje: {img_path}")
                image = Image.open(img_path).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError("Nieprawidłowy typ wejściowy. Oczekiwano ścieżki pliku lub obiektu PIL.Image.")

            # 2. Preprocessing i przeniesienie na urządzenie (GPU/CPU)
            processed_img = self.preprocess(image).unsqueeze(0).to(self.device)

            # 3. Wnioskowanie (inferencja) i normalizacja wektora
            with torch.no_grad():
                img_emb = self.model.encode_image(processed_img)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

            logger.debug("Wygenerowano embedding obrazu.")

            # 4. Konwersja tensora na standardową listę Pythona
            return img_emb.squeeze(0).cpu().tolist()

        except FileNotFoundError as e:
            logger.error(f"Nie odnaleziono pliku: {e}")
            raise ImageProcessingError(f"Błąd odczytu pliku: {e}") from e
        except Exception as e:
            logger.error(f"Błąd podczas generowania embeddingu obrazu: {e}")
            raise ImageProcessingError(f"Błąd przetwarzania obrazu: {e}") from e
    def embed_via_cloud_api(self, prompt: str) -> List[float]:

        if not self.allow_cloud:
            logger.warning("Próba użycia modelu chmurowego przy zablokowanej fladze allow_cloud=False.")
            raise CloudModelDisabledError(
                "Wywołanie modelu chmurowego jest zablokowane. "
                "Ustaw `allow_cloud=True` podczas inicjalizacji komponentu. (funkcja w fazie tworzenia)"
            )
    def get_metadata(
            self, 
            file_path: Optional[Union[str, Path]] = None, 
            execution_time_ms: float = 0.0
        ) -> Dict[str, Any]:
            """
            Generuje spójny słownik metadanych gotowy do zapisu w bazie danych.
            """
            file_name = Path(file_path).name if file_path else None

            return {
                "file_name": file_name,
                "model_name": f"OpenCLIP {self.model_name}",
                "representation_type": self.representation_type,
                "embedding_dim": self.embedding_dim,
                "execution_time_ms": round(execution_time_ms, 2),
                "timestamp": time.time()
            }
