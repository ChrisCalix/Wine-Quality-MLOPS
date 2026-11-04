"""Servicio FastAPI para inferencia del clasificador premium."""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from api.model_loader import ModelLoader
from api.schemas import ModelInfo, Prediction, PredictRequest, PredictResponse
from src.wine.config import FEATURES

loader = ModelLoader()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_task = asyncio.create_task(asyncio.to_thread(loader.try_load))
    yield
    if not load_task.done():
        load_task.cancel()


app = FastAPI(title="Wine Premium Classifier", version="0.1.0", lifespan=lifespan)


def served_by() -> str:
    return os.getenv("POD_NAME", socket.gethostname())


def score(model: object, frame: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Clase y probabilidad premium de cada fila.

    El modelo registrado devuelve ``[P(no premium), P(premium)]``; la clase sale de
    comparar ``P(premium)`` con el umbral de la versión (tag ``decision_threshold``).
    Un modelo que devuelve la clase directamente (modelo de desarrollo) conserva su salida.
    """
    raw = np.asarray(model.predict(frame))
    if raw.ndim == 2 and raw.shape[1] == 2:
        probabilities = raw[:, 1].astype(float)
        return (probabilities >= threshold).astype(int), probabilities
    labels = raw.reshape(len(frame), -1)[:, 0].astype(int)
    if hasattr(model, "predict_proba"):
        return labels, np.asarray(model.predict_proba(frame))[:, 1].astype(float)
    return labels, labels.astype(float)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if not loader.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=loader.error or "model not loaded",
        )
    return {"status": "ready"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if not loader.ready:
        raise HTTPException(status_code=503, detail=loader.error or "model not loaded")
    return ModelInfo(**loader.metadata())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not loader.ready or loader.loaded is None:
        raise HTTPException(status_code=503, detail=loader.error or "model not loaded")

    frame = pd.DataFrame([record.model_dump() for record in request.records])
    frame = frame.loc[:, list(FEATURES)].astype(float)
    labels, probabilities = score(loader.loaded.model, frame, loader.loaded.decision_threshold)
    output = [
        Prediction(premium=int(label), probability_premium=float(probability))
        for label, probability in zip(labels, probabilities, strict=True)
    ]

    metadata = loader.metadata()
    return PredictResponse(predictions=output, **{k: metadata[k] for k in (
        "model_name", "model_version", "model_alias", "run_id", "served_by"
    )})

