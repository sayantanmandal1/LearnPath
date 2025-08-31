# AI Career Recommender - Production Deployment Guide

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Security Configuration](#security-configuration)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Performance Optimization](#performance-optimization)
8. [Backup and Recovery](#backup-and-recovery)
9. [Maintenance Procedures](#maintenance-procedures)
10. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### ✅ Infrastructure Requirements
- [ ] **Compute Resources**: Minimum 4 CPU cores, 16GB RAM per service
- [ ] **Storage**: SSD storage with 1000+ IOPS, 500GB+ available space
- [ ] **Network**: Load balancer with SSL termination capability
- [ ] **Database**: PostgreSQL 14+ with replication support
- [ ] **Cache**: Redis 6+ cluster for high availability
- [ ] **Container Runtime**: Docker 20+ or Kubernetes 1.20+

### ✅ Security Requirements
- [ ] **SSL Certificates**: Valid SSL certificates for all domains
- [ ] **Firewall Rules**: Properly configured network security groups
- [ ] **Secrets Management**: Secure storage for API keys and passwords
- [ ] **Access Control**: IAM roles and permissions configured
- [ ] **Backup Encryption**: Encrypted backup storage configured

### ✅ External Dependencies
- [ ] **GitHub API**: OAuth app configured for GitHub integration
- [ ] **LinkedIn API**: API access configured (if available)
- [ ] **Vector Database**: Pinecone or Weaviate account and API keys
- [ ] **Email Service**: SMTP or email service provider configured
- [ ] **Monitoring**: Sentry, DataDog, or similar monitoring service

### ✅ Environment Preparation
- [ ] **Domain Names**: DNS records configured
- [ ] **CDN**: Content delivery network setup (optional)
- [ ] **Logging**: Centralized logging solution configured
- [ ] **Backup Storage**: S3 or equivalent backup storage
- [ ] **CI/CD Pipeline**: Automated deployment pipeline configured

## Infrastructure Setup

### Cloud Provider Setup (AWS Example)

#### 1. VPC and Networking
```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=career-recommender-vpc}]'

# Create subnets
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.1.0/24 --availability-zone us-west-2a
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.2.0/24 --availability-zone us-west-2b

# Create internet gateway
aws ec2 create-internet-gateway --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=career-recommender-igw}]'

# Attach internet gateway to VPC
aws ec2 attach-internet-gateway --vpc-id vpc-12345678 --internet-gateway-id igw-87654321
```

#### 2. Security Groups
```bash
# Create security group for web tier
aws ec2 create-security-group \
  --group-name career-recommender-web \
  --description "Security group for web tier" \
  --vpc-id vpc-12345678

# Allow HTTP/HTTPS traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-web123 \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-web123 \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Create security group for application tier
aws ec2 create-security-group \
  --group-name career-recommender-app \
  --description "Security group for application tier" \
  --vpc-id vpc-12345678

# Allow traffic from web tier
aws ec2 authorize-security-group-ingress \
  --group-id sg-app123 \
  --protocol tcp \
  --port 8000 \
  --source-group sg-web123
```

#### 3. RDS Database Setup
```bash
# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name career-recommender-db-subnet \
  --db-subnet-group-description "Subnet group for career recommender database" \
  --subnet-ids subnet-12345678 subnet-87654321

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier career-recommender-db \
  --db-instance-class db.t3.large \
  --engine postgres \
  --engine-version 14.9 \
  --master-username dbadmin \
  --master-user-password SecurePassword123! \
  --allocated-storage 100 \
  --storage-type gp2 \
  --storage-encrypted \
  --vpc-security-group-ids sg-db123 \
  --db-subnet-group-name career-recommender-db-subnet \
  --backup-retention-period 7 \
  --multi-az \
  --auto-minor-version-upgrade
```

#### 4. ElastiCache Redis Setup
```bash
# Create Redis subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name career-recommender-redis-subnet \
  --cache-subnet-group-description "Subnet group for Redis cluster" \
  --subnet-ids subnet-12345678 subnet-87654321

# Create Redis cluster
aws elasticache create-replication-group \
  --replication-group-id career-recommender-redis \
  --description "Redis cluster for career recommender" \
  --node-type cache.t3.medium \
  --engine redis \
  --engine-version 6.2 \
  --num-cache-clusters 2 \
  --cache-parameter-group-name default.redis6.x \
  --cache-subnet-group-name career-recommender-redis-subnet \
  --security-group-ids sg-redis123 \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled
```

### Kubernetes Cluster Setup

#### 1. EKS Cluster Creation
```bash
# Create EKS cluster
eksctl create cluster \
  --name career-recommender \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name career-recommender
```

#### 2. Install Required Add-ons
```bash
# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=career-recommender \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Install Ingress NGINX
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

## Security Configuration

### 1. SSL/TLS Setup

#### Let's Encrypt with cert-manager
```yaml
# cert-manager-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@career-recommender.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

#### SSL Certificate Configuration
```yaml
# ssl-certificate.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: career-recommender-tls
  namespace: default
spec:
  secretName: career-recommender-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.career-recommender.com
  - app.career-recommender.com
```

### 2. Secrets Management

#### Kubernetes Secrets
```bash
# Create database secret
kubectl create secret generic db-credentials \
  --from-literal=username=dbadmin \
  --from-literal=password=SecurePassword123! \
  --from-literal=host=career-recommender-db.cluster-xyz.us-west-2.rds.amazonaws.com \
  --from-literal=database=career_recommender

# Create API keys secret
kubectl create secret generic api-keys \
  --from-literal=github-client-id=your-github-client-id \
  --from-literal=github-client-secret=your-github-client-secret \
  --from-literal=pinecone-api-key=your-pinecone-api-key \
  --from-literal=jwt-secret-key=your-jwt-secret-key

# Create Redis secret
kubectl create secret generic redis-credentials \
  --from-literal=host=career-recommender-redis.abc123.cache.amazonaws.com \
  --from-literal=port=6379 \
  --from-literal=password=RedisPassword123!
```

#### AWS Secrets Manager Integration
```yaml
# secrets-store-csi.yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: career-recommender-secrets
spec:
  provider: aws
  parameters:
    objects: |
      - objectName: "career-recommender/db-credentials"
        objectType: "secretsmanager"
        jmesPath:
          - path: "username"
            objectAlias: "db-username"
          - path: "password"
            objectAlias: "db-password"
```

### 3. Network Security

#### Network Policies
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: career-recommender-network-policy
spec:
  podSelector:
    matchLabels:
      app: career-recommender-backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS
```

## Database Setup

### 1. Database Initialization

#### Create Production Database
```sql
-- Connect as superuser
CREATE DATABASE career_recommender;
CREATE USER career_app WITH PASSWORD 'SecureAppPassword123!';

-- Grant permissions
GRANT CONNECT ON DATABASE career_recommender TO career_app;
GRANT USAGE ON SCHEMA public TO career_app;
GRANT CREATE ON SCHEMA public TO career_app;

-- Create read-only user for analytics
CREATE USER career_readonly WITH PASSWORD 'ReadOnlyPassword123!';
GRANT CONNECT ON DATABASE career_recommender TO career_readonly;
GRANT USAGE ON SCHEMA public TO career_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO career_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO career_readonly;
```

#### Run Database Migrations
```bash
# Set database URL
export DATABASE_URL="postgresql+asyncpg://career_app:SecureAppPassword123!@career-recommender-db.cluster-xyz.us-west-2.rds.amazonaws.com:5432/career_recommender"

# Run migrations
cd backend
alembic upgrade head

# Verify migration
alembic current
alembic history
```

### 2. Database Configuration

#### PostgreSQL Configuration
```sql
-- Performance tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Connection settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Reload configuration
SELECT pg_reload_conf();
```

#### Database Monitoring Setup
```sql
-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create monitoring user
CREATE USER monitoring WITH PASSWORD 'MonitoringPassword123!';
GRANT pg_monitor TO monitoring;
```

### 3. Database Backup Configuration

#### Automated Backup Script
```bash
#!/bin/bash
# backup-database.sh

DB_HOST="career-recommender-db.cluster-xyz.us-west-2.rds.amazonaws.com"
DB_NAME="career_recommender"
DB_USER="career_app"
BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/career_recommender_$DATE.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME -f $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_FILE.gz s3://career-recommender-backups/postgresql/

# Clean up local backups older than 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

## Application Deployment

### 1. Container Images

#### Backend Dockerfile
```dockerfile
# backend/Dockerfile.prod
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Frontend Dockerfile
```dockerfile
# frontend/Dockerfile.prod
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

### 2. Kubernetes Deployments

#### Backend Deployment
```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: career-recommender-backend
  labels:
    app: career-recommender-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: career-recommender-backend
  template:
    metadata:
      labels:
        app: career-recommender-backend
    spec:
      containers:
      - name: backend
        image: career-recommender-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: jwt-secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: career-recommender-backend-service
spec:
  selector:
    app: career-recommender-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Frontend Deployment
```yaml
# k8s/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: career-recommender-frontend
  labels:
    app: career-recommender-frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: career-recommender-frontend
  template:
    metadata:
      labels:
        app: career-recommender-frontend
    spec:
      containers:
      - name: frontend
        image: career-recommender-frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NEXT_PUBLIC_API_URL
          value: "https://api.career-recommender.com"
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.5"
          limits:
            memory: "1Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: career-recommender-frontend-service
spec:
  selector:
    app: career-recommender-frontend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: ClusterIP
```

#### Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: career-recommender-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.career-recommender.com
    - app.career-recommender.com
    secretName: career-recommender-tls
  rules:
  - host: api.career-recommender.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: career-recommender-backend-service
            port:
              number: 80
  - host: app.career-recommender.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: career-recommender-frontend-service
            port:
              number: 80
```

### 3. Celery Worker Deployment

#### Celery Worker Configuration
```yaml
# k8s/celery-worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: career-recommender-celery-worker
  labels:
    app: career-recommender-celery-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: career-recommender-celery-worker
  template:
    metadata:
      labels:
        app: career-recommender-celery-worker
    spec:
      containers:
      - name: celery-worker
        image: career-recommender-backend:latest
        command: ["celery", "-A", "app.core.celery_app", "worker", "--loglevel=info"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
```

## Monitoring Setup

### 1. Application Monitoring

#### Prometheus Configuration
```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'career-recommender-backend'
      static_configs:
      - targets: ['career-recommender-backend-service:80']
      metrics_path: /metrics
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Career Recommender Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### 2. Log Aggregation

#### Fluentd Configuration
```yaml
# logging/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      read_from_head true
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name career-recommender
      type_name _doc
    </match>
```

### 3. Health Checks

#### Custom Health Check Endpoint
```python
# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from app.core.database import get_db
from app.core.redis import get_redis
from app.services.health_service import HealthService

router = APIRouter()

@router.get("/health/")
async def health_check():
    """Comprehensive health check"""
    health_service = HealthService()
    
    health_status = await health_service.check_system_health()
    
    return {
        "status": health_status["overall_status"],
        "timestamp": health_status["timestamp"],
        "components": health_status["components"],
        "version": "1.0.0"
    }

@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    # Check if application is ready to serve traffic
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    # Check if application is alive
    return {"status": "alive"}
```

## Performance Optimization

### 1. Database Optimization

#### Connection Pooling
```python
# app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Optimized connection pool settings
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)
```

#### Query Optimization
```sql
-- Add performance indexes
CREATE INDEX CONCURRENTLY idx_jobs_skills_gin ON jobs USING gin(required_skills);
CREATE INDEX CONCURRENTLY idx_profiles_updated_at ON profiles(updated_at DESC);
CREATE INDEX CONCURRENTLY idx_recommendations_score ON recommendations(compatibility_score DESC);

-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM jobs 
WHERE required_skills @> '["Python"]'::jsonb 
ORDER BY posted_date DESC 
LIMIT 10;
```

### 2. Caching Strategy

#### Redis Caching Configuration
```python
# app/services/cache_service.py
import redis.asyncio as redis
from typing import Optional, Any
import json

class CacheService:
    def __init__(self):
        self.redis = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True
        )
    
    async def get_or_set(self, key: str, fetch_func: callable, ttl: int = 3600) -> Any:
        """Get from cache or fetch and cache"""
        cached_value = await self.redis.get(key)
        
        if cached_value:
            return json.loads(cached_value)
        
        # Fetch fresh data
        fresh_data = await fetch_func()
        
        # Cache the result
        await self.redis.setex(key, ttl, json.dumps(fresh_data, default=str))
        
        return fresh_data
```

### 3. Application Performance

#### Async Optimization
```python
# app/services/recommendation_service.py
import asyncio
from typing import List

class RecommendationService:
    async def get_recommendations_parallel(self, profile_id: str) -> Dict:
        """Get recommendations using parallel processing"""
        
        # Run multiple recommendation algorithms in parallel
        tasks = [
            self.collaborative_filtering(profile_id),
            self.content_based_filtering(profile_id),
            self.skill_based_matching(profile_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        return self.combine_recommendations(results)
```

## Backup and Recovery

### 1. Database Backup Strategy

#### Automated Backup CronJob
```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | gzip > /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz
              aws s3 cp /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz s3://career-recommender-backups/
            env:
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: host
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: username
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: password
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

### 2. Application State Backup

#### Redis Backup Script
```bash
#!/bin/bash
# backup-redis.sh

REDIS_HOST="career-recommender-redis.abc123.cache.amazonaws.com"
BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Redis data
redis-cli -h $REDIS_HOST --rdb $BACKUP_DIR/redis-backup-$DATE.rdb

# Compress backup
gzip $BACKUP_DIR/redis-backup-$DATE.rdb

# Upload to S3
aws s3 cp $BACKUP_DIR/redis-backup-$DATE.rdb.gz s3://career-recommender-backups/redis/

echo "Redis backup completed: redis-backup-$DATE.rdb.gz"
```

### 3. Disaster Recovery Plan

#### Recovery Procedures
```bash
#!/bin/bash
# disaster-recovery.sh

echo "Starting disaster recovery process..."

# 1. Restore database from backup
echo "Restoring database..."
LATEST_BACKUP=$(aws s3 ls s3://career-recommender-backups/postgresql/ | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://career-recommender-backups/postgresql/$LATEST_BACKUP /tmp/
gunzip /tmp/$LATEST_BACKUP
psql -h $NEW_DB_HOST -U $DB_USER -d $DB_NAME -f /tmp/${LATEST_BACKUP%.gz}

# 2. Restore Redis data
echo "Restoring Redis data..."
LATEST_REDIS_BACKUP=$(aws s3 ls s3://career-recommender-backups/redis/ | sort | tail -n 1 | awk '{print $4}')
aws s3 cp s3://career-recommender-backups/redis/$LATEST_REDIS_BACKUP /tmp/
gunzip /tmp/$LATEST_REDIS_BACKUP
redis-cli -h $NEW_REDIS_HOST --pipe < /tmp/${LATEST_REDIS_BACKUP%.gz}

# 3. Update DNS records
echo "Updating DNS records..."
aws route53 change-resource-record-sets --hosted-zone-id $HOSTED_ZONE_ID --change-batch file://dns-update.json

# 4. Deploy application
echo "Deploying application..."
kubectl apply -f k8s/

echo "Disaster recovery completed!"
```

## Maintenance Procedures

### 1. Regular Maintenance Tasks

#### Weekly Maintenance Script
```bash
#!/bin/bash
# weekly-maintenance.sh

echo "Starting weekly maintenance..."

# 1. Database maintenance
echo "Running database maintenance..."
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "VACUUM ANALYZE;"
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "REINDEX DATABASE career_recommender;"

# 2. Clean up old logs
echo "Cleaning up old logs..."
kubectl delete pods --field-selector=status.phase==Succeeded -A

# 3. Update container images
echo "Updating container images..."
kubectl set image deployment/career-recommender-backend backend=career-recommender-backend:latest
kubectl set image deployment/career-recommender-frontend frontend=career-recommender-frontend:latest

# 4. Restart services if needed
echo "Checking service health..."
kubectl rollout status deployment/career-recommender-backend
kubectl rollout status deployment/career-recommender-frontend

echo "Weekly maintenance completed!"
```

### 2. Security Updates

#### Security Patch Process
```bash
#!/bin/bash
# security-updates.sh

echo "Applying security updates..."

# 1. Update base images
docker pull python:3.12-slim
docker pull node:18-alpine
docker pull postgres:14
docker pull redis:7-alpine

# 2. Rebuild application images
docker build -t career-recommender-backend:security-patch ./backend
docker build -t career-recommender-frontend:security-patch ./frontend

# 3. Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image career-recommender-backend:security-patch

# 4. Deploy if scan passes
if [ $? -eq 0 ]; then
    kubectl set image deployment/career-recommender-backend backend=career-recommender-backend:security-patch
    kubectl set image deployment/career-recommender-frontend frontend=career-recommender-frontend:security-patch
    echo "Security updates applied successfully!"
else
    echo "Security scan failed! Manual review required."
    exit 1
fi
```

## Troubleshooting

### 1. Common Issues and Solutions

#### Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it deployment/career-recommender-backend -- psql $DATABASE_URL -c "SELECT 1;"

# Check connection pool status
kubectl logs deployment/career-recommender-backend | grep "pool"

# Restart database connections
kubectl rollout restart deployment/career-recommender-backend
```

#### Redis Connection Issues
```bash
# Check Redis connectivity
kubectl exec -it deployment/career-recommender-backend -- redis-cli -u $REDIS_URL ping

# Check Redis memory usage
kubectl exec -it deployment/career-recommender-backend -- redis-cli -u $REDIS_URL info memory

# Clear Redis cache if needed
kubectl exec -it deployment/career-recommender-backend -- redis-cli -u $REDIS_URL flushall
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods
kubectl top nodes

# Check application metrics
curl -s http://api.career-recommender.com/metrics | grep http_requests

# Scale up if needed
kubectl scale deployment career-recommender-backend --replicas=5
```

### 2. Monitoring and Alerting

#### Critical Alerts Configuration
```yaml
# alerts/critical-alerts.yaml
groups:
- name: career-recommender-critical
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: DatabaseDown
    expr: up{job="postgresql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database is down"
      description: "PostgreSQL database is not responding"

  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90%"
```

### 3. Emergency Procedures

#### Emergency Rollback
```bash
#!/bin/bash
# emergency-rollback.sh

echo "Initiating emergency rollback..."

# 1. Rollback to previous deployment
kubectl rollout undo deployment/career-recommender-backend
kubectl rollout undo deployment/career-recommender-frontend

# 2. Wait for rollback to complete
kubectl rollout status deployment/career-recommender-backend --timeout=300s
kubectl rollout status deployment/career-recommender-frontend --timeout=300s

# 3. Verify health
curl -f http://api.career-recommender.com/api/v1/health/ || exit 1

echo "Emergency rollback completed successfully!"
```

#### Scale Down for Maintenance
```bash
#!/bin/bash
# maintenance-mode.sh

echo "Entering maintenance mode..."

# 1. Scale down to single replica
kubectl scale deployment career-recommender-backend --replicas=1
kubectl scale deployment career-recommender-frontend --replicas=1

# 2. Update ingress to show maintenance page
kubectl apply -f k8s/maintenance-ingress.yaml

# 3. Wait for maintenance completion
read -p "Press enter when maintenance is complete..."

# 4. Scale back up
kubectl scale deployment career-recommender-backend --replicas=3
kubectl scale deployment career-recommender-frontend --replicas=2

# 5. Restore normal ingress
kubectl apply -f k8s/ingress.yaml

echo "Maintenance mode completed!"
```

---

## Post-Deployment Checklist

### ✅ Verification Steps
- [ ] **Application Health**: All health checks passing
- [ ] **Database Connectivity**: Database connections working
- [ ] **Cache Functionality**: Redis cache operational
- [ ] **External APIs**: Third-party integrations working
- [ ] **SSL Certificates**: HTTPS working correctly
- [ ] **Monitoring**: All monitoring systems active
- [ ] **Backup Systems**: Automated backups running
- [ ] **Performance**: Response times within SLA
- [ ] **Security**: Security scans completed
- [ ] **Documentation**: Deployment documented

### 🚀 Go-Live Activities
1. **DNS Cutover**: Update DNS to point to new infrastructure
2. **Traffic Monitoring**: Monitor traffic patterns and performance
3. **Error Monitoring**: Watch for any error spikes
4. **Performance Validation**: Verify performance meets requirements
5. **User Communication**: Notify users of successful deployment
6. **Team Notification**: Inform development and operations teams

This deployment guide provides a comprehensive approach to deploying the AI Career Recommender system in a production environment with high availability, security, and performance considerations.