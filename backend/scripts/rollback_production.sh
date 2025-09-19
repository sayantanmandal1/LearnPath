#!/bin/bash
# Production Rollback Script for AI Career Recommender

set -e

echo "🔄 Starting AI Career Recommender Production Rollback"
echo "=================================================="

COMPOSE_FILE="docker-compose.production.yml"
BACKUP_DIR="${1:-./backups/latest}"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    echo "Usage: $0 [backup_directory]"
    exit 1
fi

echo "📁 Using backup directory: $BACKUP_DIR"

# Stop current services
echo "🛑 Stopping current services..."
docker-compose -f "$COMPOSE_FILE" down

# Restore database backup
if [ -f "$BACKUP_DIR/database_backup.sql" ]; then
    echo "🗄️ Restoring database backup..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres
    sleep 10
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U ai_career_user -d ai_career_db < "$BACKUP_DIR/database_backup.sql"
fi

# Start services with previous configuration
echo "🚀 Starting services with rollback configuration..."
docker-compose -f "$COMPOSE_FILE" up -d

echo "✅ Rollback completed successfully!"
