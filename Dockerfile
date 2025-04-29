# Use an official lightweight Python image as the base
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set a working directory inside the container
WORKDIR /app

# Copy your project files into the container (adjust as needed)
COPY . .

# Install system-level dependencies for PyTorch, datasets, and plotting
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# It's best practice to use a requirements.txt, but for clarity, we list them here.
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    sentence-transformers \
    datasets \
    pytorch-optimizer \
    matplotlib \
    scipy

# Download the Hugging Face model weights at build time to avoid runtime downloads
# This ensures the model is cached in the Docker image and not re-downloaded each run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Expose a port if you plan to serve the model via an API (optional)
# EXPOSE 5000

# Set the default command to run your main script (adjust filename as needed)
CMD ["python", "multitask_learning_sentence_transformer.py"]
