from pydantic import BaseModel, field_validator
from typing import List
import numpy as np

class PredictionRequest(BaseModel):
    features: List[float]
    
    @field_validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Features must contain exactly 4 values')
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        if any(x < 0 for x in v):
            raise ValueError('All features must be non-negative')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    prediction_name: str
    confidence: List[float]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

    
    
