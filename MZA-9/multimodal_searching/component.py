import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import open_clip
from PIL import Image

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
        self.model_name = model_name
        self.pretrained = pretrained
        self.allow_cloud = allow_cloud
        self.representation_type = representation_type

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Inicjalizacja | Model: {self.model_name} | Urządzenie: {self.device} | Cloud: {self.allow_cloud}"
        )

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()

            # Dynamiczne ustalanie wymiaru embeddingu (Punkt 4)
            with torch.no_grad():
                _dummy = self.tokenizer(["x"]).to(self.device)
                self.embedding_dim = self.model.encode_text(_dummy).shape[-1]

            logger.info(f"Model załadowany. Wymiar embeddingu: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Nie udało się załadować modelu: {e}")
            raise EmbeddingError(f"Błąd inicjalizacji modelu: {e}") from e

    def embed_text(self, text: str) -> List[float]:
        # Walidacja danych z własnym wyjątkiem (Punkt 5)
        if not text or not text.strip():
            logger.error("Próba przetworzenia pustego tekstu.")
            raise EmbeddingError("Tekst do wektoryzacji nie może być pusty.")

        try:
            tokens = self.tokenizer([text]).to(self.device)

            with torch.no_grad():
                text_emb = self.model.encode_text(tokens)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)

            logger.debug(f"Wygenerowano embedding tekstu dla: '{text[:20]}...'")
            return text_emb.squeeze(0).cpu().tolist()

        except Exception as e:
            logger.error(f"Błąd podczas generowania embeddingu tekstu: {e}")
            raise EmbeddingError(f"Błąd przetwarzania tekstu: {e}") from e

    def embed_image(self, image_input: Union[str, Path, Image.Image]) -> List[float]:
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    raise FileNotFoundError(f"Plik obrazu nie istnieje: {img_path}")
                image = Image.open(img_path).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ImageProcessingError(
                    "Nieprawidłowy typ wejściowy. Oczekiwano ścieżki pliku lub obiektu PIL.Image."
                )

            processed_img = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_emb = self.model.encode_image(processed_img)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

            logger.debug("Wygenerowano embedding obrazu.")
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
                "Ustaw `allow_cloud=True` podczas inicjalizacji komponentu."
            )

    def get_metadata(
        self, 
        file_path: Optional[Union[str, Path]] = None, 
        execution_time_ms: float = 0.0
    ) -> Dict[str, Any]:
        file_name = Path(file_path).name if file_path else None

        return {
            "file_name": file_name,
            "model_name": f"{self.model_name}",
            "representation_type": self.representation_type,
            "embedding_dim": self.embedding_dim,
            "execution_time_ms": round(execution_time_ms, 2),
            "timestamp": time.time()
        }