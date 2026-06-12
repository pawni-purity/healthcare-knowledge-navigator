import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator, List
from unittest.mock import MagicMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.database import get_db
from backend.app.db.models import Base
from backend.app.core.config import settings
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.indexer import IndexerService

# Force OpenAI provider for test suite mock verification
settings.LLM_PROVIDER = "openai"

# Use an in-memory SQLite database for testing, running in async mode
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """
    Creates an instance of the default event loop for the test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a transactional session per test, rolling back mutations at completion.
    """
    AsyncSessionTesting = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
    async with AsyncSessionTesting() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

# Mock Embeddings fixture
@pytest.fixture(autouse=True)
def mock_embeddings():
    """
    Mocks EmbeddingService calls to prevent downloading BGE model during unit testing.
    Returns dummy 1024-dimensional vectors.
    """
    def dummy_embed_text(text: str):
        return [0.1] * 1024

    def dummy_embed_batch(texts: List[str]):
        return [[0.1] * 1024 for _ in texts]

    with patch.object(EmbeddingService, "embed_text", side_effect=dummy_embed_text), \
         patch.object(EmbeddingService, "embed_batch", side_effect=dummy_embed_batch), \
         patch.object(EmbeddingService, "get_model", return_value=MagicMock()):
        yield

# Mock Qdrant client fixture
@pytest.fixture(autouse=True)
def mock_qdrant():
    """
    Mocks QdrantClient connections and operations.
    """
    mock_client = MagicMock()
    # Mock return values for collection checks
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections

    with patch("backend.app.services.indexer.QdrantClient", return_value=mock_client), \
         patch("backend.app.services.search.QdrantClient", return_value=mock_client):
        yield mock_client

@pytest.fixture
def client(db_session, test_engine) -> TestClient:
    """
    Prepares a test API client routing database queries to the transactional SQLite mock session.
    """
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    
    # Redirect lifespan engine begin calls to SQLite test engine
    import backend.app.main
    original_engine = backend.app.main.engine
    backend.app.main.engine = test_engine
    
    # Prevent client lifespan from disposing the shared test session engine
    from sqlalchemy.ext.asyncio import AsyncEngine
    from unittest.mock import AsyncMock, patch
    with patch.object(AsyncEngine, "dispose", AsyncMock(return_value=None)):
        try:
            with TestClient(app) as test_client:
                yield test_client
        finally:
            backend.app.main.engine = original_engine
            app.dependency_overrides.clear()
