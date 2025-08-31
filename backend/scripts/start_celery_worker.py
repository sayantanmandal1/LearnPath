#!/usr/bin/env python3
"""
Celery worker startup script
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.celery_app import celery_app

if __name__ == "__main__":
    # Start Celery worker
    celery_app.start(argv=[
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--queues=default,ml_queue,data_queue,cache_queue,analytics_queue",
        "--hostname=worker@%h"
    ])