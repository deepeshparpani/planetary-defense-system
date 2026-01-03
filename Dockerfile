# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Cloud Run expects the application to listen on the port defined by the PORT env variable
ENV PORT 8080

# The command depends on whether you are launching the backend or frontend
# We will use a script or environment variable to toggle this in Cloud Run
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT