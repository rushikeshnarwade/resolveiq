# Use an official, lightweight Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for psycopg (PostgreSQL driver)
RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}