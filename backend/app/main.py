import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.api.router import api_router
from backend.app.db.models import Base
from backend.app.core.database import engine

# Configure logger formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Triggers startup and shutdown routines.
    Auto-creates database tables using SQLAlchemy Declarative definitions.
    """
    logger.info("Initializing relational database tables...")
    async with engine.begin() as conn:
        # Create all tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")
    yield
    # Clean up on shutdown
    logger.info("Shutting down database connections...")
    await engine.dispose()

app = FastAPI(
    title="Healthcare Knowledge Navigator API",
    description="Backend API for document ingestion, processing, and vector indexing.",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS middleware rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect global API routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["health"])
async def health_check():
    """
    Standard heartbeat endpoint for container orchestrators.
    """
    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "qdrant_host": settings.QDRANT_HOST
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
