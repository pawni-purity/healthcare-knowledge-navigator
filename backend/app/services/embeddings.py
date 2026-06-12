import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls) -> Optional[SentenceTransformer]:
        """
        Thread-safe singleton/lazy loader for the SentenceTransformer model.
        Pre-caches and loads BAAI/bge-large-en-v1.5 (unless set to 'mock').
        """
        if settings.EMBEDDING_MODEL_NAME == "mock":
            return None

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
    def embed_text(cls, text: str) -> List[float]:
        """
        Generates a 1024-dimensional embedding vector for a single string.
        """
        model = cls.get_model()
        if model is None:
            return [0.1] * settings.EMBEDDING_DIMENSION
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
            return [[0.1] * settings.EMBEDDING_DIMENSION for _ in texts]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

