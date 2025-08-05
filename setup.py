from setuptools import setup, find_packages

setup(
    name="iris-mlops",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "mlflow>=2.7.1",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "pydantic>=2.3.0",
        "prometheus-client>=0.17.1",
        "pytest>=7.4.2",
        "joblib>=1.3.2"
    ],
    author="Rajat Jamblekar",
    description="MLOps pipeline for Iris classification",
    python_requires=">=3.8",
)