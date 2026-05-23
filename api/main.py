"""
AERO.AI MVP1 — main.py
FastAPI application entry point.

Startup (lifespan):
  - Loads ONNX session once into memory
  - Loads all 6 scalers once into memory
  - Any failure here aborts startup cleanly — no silent bad state

Run:
  cd api/
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Cloudflare tunnel (separate terminal):
  cloudflared tunnel --url http://localhost:8000
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers.predict import router as predict_router
from services.inference import load_model

log = structlog.get_logger()


# ------------------------------------------------------------------ #
#  Lifespan — runs once on startup, once on shutdown                  #
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    log.info("startup", msg="Loading ONNX model and scalers...")
    try:
        load_model()
        log.info("startup", msg="✅ Model loaded successfully", model=settings.model_version)
    except FileNotFoundError as e:
        log.error("startup_failed", error=str(e))
        raise RuntimeError(f"Startup aborted — {e}") from e

    yield  # app is running

    # --- SHUTDOWN ---
    log.info("shutdown", msg="AERO.AI API shutting down.")


# ------------------------------------------------------------------ #
#  App                                                                #
# ------------------------------------------------------------------ #
app = FastAPI(
    title       = "AERO.AI MVP1",
    description = (
        "Aerodynamic coefficient prediction API for NACA 4-digit airfoils. "
        "Powered by MultiFidelityAeroAI (ONNX) with MC Dropout uncertainty."
    ),
    version     = settings.api_version,
    lifespan    = lifespan,
)


# ------------------------------------------------------------------ #
#  CORS — locked to Vercel dashboard (Session F)                      #
# ------------------------------------------------------------------ #
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["https://aeroai-dashboard.vercel.app"],
    allow_methods  = ["POST", "GET"],
    allow_headers  = ["*"],
)


# ------------------------------------------------------------------ #
#  Routers                                                            #
# ------------------------------------------------------------------ #
app.include_router(predict_router)


# ------------------------------------------------------------------ #
#  Health check — used to verify API is alive after CF tunnel starts  #
# ------------------------------------------------------------------ #
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   settings.model_version,
        "version": settings.api_version,
    }


# ------------------------------------------------------------------ #
#  Root                                                               #
# ------------------------------------------------------------------ #
@app.get("/")
def root():
    return {
        "name":    "AERO.AI MVP1",
        "docs":    "/docs",
        "predict": f"/{settings.api_version}/predict",
        "health":  "/health",
    }
