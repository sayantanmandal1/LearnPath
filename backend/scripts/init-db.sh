#!/bin/bash

# Database initialization script

set -e

echo "Initializing database..."

# Create initial migration
echo "Creating initial migration..."
alembic revision --autogenerate -m "Initial migration"

# Run migrations
echo "Running migrations..."
alembic upgrade head

echo "Database initialization complete!"