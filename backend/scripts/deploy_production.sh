#!/bin/bash
# Production Deployment Script for AI Career Recommender

set -e

echo "🚀 Starting AI Career Recommender Production Deployment"
echo "=================================================="

# Configuration
COMPOSE_FILE="docker-compose.production.yml"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if required files exist
required_files=(
    "backend/.env.production"
    "docker-compose.production.yml"
    "nginx/nginx.conf"
    "monitoring/prometheus.yml"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files present"

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create backup directory
mkdir -p "$BACKUP_DIR"
echo "📁 Created backup directory: $BACKUP_DIR"

# Backup current deployment (if exists)
if [ -f "$COMPOSE_FILE" ]; then
    echo "💾 Creating backup of current deployment..."
    docker-compose -f "$COMPOSE_FILE" exec postgres pg_dump -U ai_career_user ai_career_db > "$BACKUP_DIR/database_backup.sql" || true
    docker-compose -f "$COMPOSE_FILE" logs > "$BACKUP_DIR/application_logs.txt" || true
fi

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose -f "$COMPOSE_FILE" pull

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose -f "$COMPOSE_FILE" down || true

# Start new deployment
echo "🚀 Starting new deployment..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Health checks
echo "🏥 Running health checks..."
services=("backend" "frontend" "postgres" "redis")

for service in "${services[@]}"; do
    if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up (healthy)"; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service is not healthy"
        docker-compose -f "$COMPOSE_FILE" logs "$service"
        exit 1
    fi
done

# Run database migrations
echo "🗄️ Running database migrations..."
docker-compose -f "$COMPOSE_FILE" exec backend alembic upgrade head

# Verify API endpoints
echo "🌐 Verifying API endpoints..."
if curl -f http://localhost:8000/api/v1/health/ > /dev/null 2>&1; then
    echo "✅ Backend API is responding"
else
    echo "❌ Backend API is not responding"
    exit 1
fi

if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is responding"
else
    echo "❌ Frontend is not responding"
    exit 1
fi

# Final status
echo ""
echo "🎉 Deployment completed successfully!"
echo "=================================================="
echo "📊 Service Status:"
docker-compose -f "$COMPOSE_FILE" ps

echo ""
echo "🔗 Access URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Monitoring: http://localhost:9090 (Prometheus)"
echo "   Dashboards: http://localhost:3001 (Grafana)"

echo ""
echo "📋 Next Steps:"
echo "   1. Configure SSL certificates in nginx/ssl/"
echo "   2. Update domain names in nginx.conf"
echo "   3. Set up DNS records"
echo "   4. Configure monitoring alerts"
echo "   5. Set up automated backups"

echo ""
echo "✅ Production deployment is ready!"
