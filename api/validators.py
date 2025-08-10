from pydantic import BaseModel, field_validator, Field
from typing import List, Optional
import numpy as np

class EnhancedPredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="4 iris features: sepal_length, sepal_width, petal_length, petal_width",
        example=[5.1, 3.5, 1.4, 0.2]
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use for prediction"
    )
    
    @field_validator('features')
    def check_feature_count(cls, v):
        if len(v) != 4:
            raise ValueError("Exactly 4 features required (sepal_length, sepal_width, petal_length, petal_width)")
        return v
    
    @field_validator('features')
    def check_feature_values(cls, v):
        # Define acceptable ranges for each feature
        feature_ranges = [
            (0, 8),   # sepal_length (cm)
            (0, 5),   # sepal_width (cm)
            (0, 7),   # petal_length (cm)
            (0, 3)    # petal_width (cm)
        ]
        
        for i, (value, (min_val, max_val)) in enumerate(zip(v, feature_ranges)):
            if not (min_val <= value <= max_val):
                feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                raise ValueError(
                    f"Feature {feature_names[i]} must be between {min_val} and {max_val}, got {value}"
                )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "model_version": "latest"
            }
        }