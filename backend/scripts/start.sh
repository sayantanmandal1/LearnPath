#!/bin/bash

# Start script for the AI Career Recommender backend

set -e

echo "Starting AI Career Recommender Backend..."

# Wait for database to be ready
echo "Waiting for database..."
while ! nc -z localhost 5432; do
  sleep 0.1
done
echo "Database is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis..."
while ! nc -z localhost 6379; do
  sleep 0.1
done
echo "Redis is ready!"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload