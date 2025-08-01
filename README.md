# MLOps Pipeline for Iris Dataset

#steps

mkdir mlops-iris
cd mlops-iris

git init

git remote add origin https://github.com/your-username/mlops-iris.git

echo "# MLOps Pipeline for Iris Dataset" > README.md
git add README.md
git commit -m "Initial commit: Setup project repo"
git push -u origin main

mkdir -p data notebooks src api logs .github/workflows
touch requirements.txt .gitignore

python -m venv venv

.\venv\Scripts\Activate.ps1


pip install numpy pandas scikit-learn jupyter dvc mlflow flask fastapi uvicorn

run this to load and preprocess the iris dataset
python src/data_loader.py

python src/train.py

Start MLflow UI to browse experiments:
mlflow ui

Go to http://localhost:5000

Run the API locally(FASTAPI)
uvicorn api.app:app --reload

Build & Run Docker Container
Build image
docker build -t iris-api .

Run container
docker run -p 8000:8000 iris-api

Test API

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
-d '{"features": [5.1, 3.5, 1.4, 0.2]}'