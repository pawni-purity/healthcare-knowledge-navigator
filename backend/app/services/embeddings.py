import logging
import threading
import hashlib
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Optional[SentenceTransformer]:
        """
        Thread-safe singleton/lazy loader for the configured
        SentenceTransformer embedding model.
        """
        if settings.EMBEDDING_MODEL_NAME == "mock":
            return None

        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
                    try:
                        # Load sentence transformer model. On CPU or GPU automatically.
                        cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                    except Exception as e:
                        logger.error(f"Failed to load sentence transformer model: {e}")
                        raise e
        return cls._model

    @classmethod
    def _generate_mock_vector(cls, text: str) -> List[float]:
        """Generates a deterministic pseudo-random vector based on text hash."""
        # Use md5 to create a deterministic hash of the text
        hash_digest = hashlib.md5(text.encode('utf-8')).digest()
        vector = []
        for i in range(settings.EMBEDDING_DIMENSION):
            # Normalize a byte to a float between -0.5 and 0.5
            val = (hash_digest[i % len(hash_digest)] / 255.0) - 0.5
            vector.append(val)
        return vector

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """
        Generates an embedding vector for a single string.
        """
        model = cls.get_model()
        if model is None:
            return cls._generate_mock_vector(text)
        # Ensure we generate list of floats (numpy array converted)
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    @classmethod
    def embed_batch(cls, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of strings.
        """
        if not texts:
            return []
        model = cls.get_model()
        if model is None:
            return [cls._generate_mock_vector(t) for t in texts]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
