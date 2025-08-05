from pydantic import BaseModel, field_validator, Field
from typing import List, Optional
import numpy as np

class EnhancedPredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Four iris features: sepal_length, sepal_width, petal_length, petal_width",
        example=[5.1, 3.5, 1.4, 0.2]
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use for prediction"
    )
    
    @field_validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Features must contain exactly 4 values: [sepal_length, sepal_width, petal_length, petal_width]')
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for i, (feature_name, value) in enumerate(zip(feature_names, v)):
            if not isinstance(value, (int, float)):
                raise ValueError(f'{feature_name} must be numeric')
            if value < 0:
                raise ValueError(f'{feature_name} must be non-negative')
            if value > 20:  # Reasonable upper bound for iris measurements
                raise ValueError(f'{feature_name} seems too large (>{20})')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "model_version": "latest"
            }
        }