FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by some packages (chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the cross-encoder model into the image so first requests are fast
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the rest of the application
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
