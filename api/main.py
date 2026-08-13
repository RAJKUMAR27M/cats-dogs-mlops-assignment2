"""FastAPI inference service for Cats vs Dogs binary classification."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from typing import Optional

from api.predictor import Predictor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
api_logger = logging.getLogger("api.requests")

# ---------------------------------------------------------------------------
# In-app counters (lightweight alternative to full Prometheus client)
# ---------------------------------------------------------------------------
predictor: Optional[Predictor] = None
request_count: int = 0
total_latency: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan – load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    model_path = os.environ.get("MODEL_PATH", "artifacts/model/model.pt")
    logger.info("Loading model from: %s", model_path)
    predictor = Predictor(model_path=model_path)
    if predictor.is_loaded():
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded – service running in limited mode")
    yield
    logger.info("Shutting down inference service")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="MLOps inference service – binary image classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint at /metrics
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, include_in_schema=False, tags=["monitoring"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["monitoring"])
async def health_check() -> Dict[str, Any]:
    """Return service health and model status."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.is_loaded(),
        "version": "1.0.0",
        "request_count": request_count,
        "avg_latency_ms": round(
            total_latency / max(request_count, 1) * 1000, 2
        ),
    }


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict cat or dog from an uploaded image.

    - **file**: JPEG or PNG image file

    Returns predicted label, confidence, per-class probabilities, and latency.
    """
    global request_count, total_latency

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    if predictor is None or not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")

    start = time.time()
    try:
        image_bytes = await file.read()
        result = predictor.predict_from_bytes(image_bytes)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    latency = time.time() - start
    request_count += 1
    total_latency += latency

    response = {
        "filename": file.filename,
        "prediction": result["label"],
        "confidence": round(result["confidence"], 4),
        "probabilities": {
            "cat": round(1.0 - result["raw_score"], 4),
            "dog": round(result["raw_score"], 4),
        },
        "latency_ms": round(latency * 1000, 2),
    }

    # Log request without sensitive data
    api_logger.info(
        "PREDICT | file=%s | pred=%s | conf=%.4f | latency=%.1fms",
        file.filename,
        response["prediction"],
        response["confidence"],
        response["latency_ms"],
    )

    return JSONResponse(content=response)


@app.get("/stats", tags=["monitoring"])
async def get_stats() -> Dict[str, Any]:
    """Return basic API usage statistics."""
    return {
        "total_requests": request_count,
        "avg_latency_ms": round(
            total_latency / max(request_count, 1) * 1000, 2
        ),
        "total_latency_s": round(total_latency, 3),
    }


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
