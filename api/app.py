# from fastapi import FastAPI
# from pydantic import BaseModel, conlist
# import mlflow.pyfunc
# import os

# app = FastAPI(title="Iris Model API")

# # Load model from MLflow Model Registry
# MODEL_NAME = "iris_model"
# MODEL_STAGE = "Production"

# model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# class IrisInput(BaseModel):
#     features: conlist(float, min_items=4, max_items=4)

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Iris prediction API!"}

# @app.post("/predict")
# def predict(input: IrisInput):
#     prediction = model.predict([input.features])
#     return {"prediction": int(prediction[0])}

from fastapi import FastAPI
from pydantic import BaseModel, conlist
import mlflow.pyfunc
import os

app = FastAPI(title="Iris Model API")

# mlflow.set_tracking_uri("file:/app/mlruns")

 # Serve static files (like favicon)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Load the latest model from MLflow Model Registry
# MODEL_NAME = "iris_model"

# # Load the model without specifying a stage
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/latest")



model = mlflow.pyfunc.load_model("model")


class IrisInput(BaseModel):
    features: conlist(float, min_length=4, max_length=4)  # Fixed parameter names

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris prediction API!"}

@app.post("/predict")
def predict(input: IrisInput):
    prediction = model.predict([input.features])
    return {"prediction": int(prediction[0])}
