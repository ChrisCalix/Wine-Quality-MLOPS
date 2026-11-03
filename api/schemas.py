"""Esquemas Pydantic del contrato HTTP de inferencia."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class WineRecord(BaseModel):
    """Una observación con las once variables fisicoquímicas del vino."""

    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(ge=0)
    ph: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0)


class PredictRequest(BaseModel):
    records: list[WineRecord] = Field(min_length=1)


class Prediction(BaseModel):
    premium: int = Field(ge=0, le=1)
    probability_premium: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_name: str
    model_version: str
    model_alias: str
    run_id: str
    served_by: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_alias: str
    run_id: str
    loaded_at: datetime
    served_by: str

