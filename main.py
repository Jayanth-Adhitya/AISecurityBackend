from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.endpoints import videos, queries, anomalies, websocket

# Basic logging configuration to surface debug/info logs from services during
# development. Adjust or remove in production.
logging.basicConfig(level=logging.INFO)
logging.getLogger("app.services.video_analyzer").setLevel(logging.DEBUG)
from contextlib import asynccontextmanager

from config import settings
from api.endpoints import videos, queries

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(videos.router, prefix="/api/v1", tags=["videos"])
app.include_router(queries.router, prefix="/api/v1", tags=["queries"])
app.include_router(anomalies.router, prefix="/api/v1", tags=["anomalies"])
app.include_router(websocket.router, tags=["websocket"])

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}