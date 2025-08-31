#!/usr/bin/env python3
"""
Celery beat (scheduler) startup script
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.celery_app import celery_app

if __name__ == "__main__":
    # Start Celery beat scheduler
    celery_app.start(argv=[
        "beat",
        "--loglevel=info",
        "--schedule=/tmp/celerybeat-schedule",
        "--pidfile=/tmp/celerybeat.pid"
    ])