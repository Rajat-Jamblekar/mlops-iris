# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# Copy all project files
COPY . .

# Copy the mlruns directory
COPY mlruns ./mlruns

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
