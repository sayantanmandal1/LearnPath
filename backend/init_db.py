#!/usr/bin/env python3
"""
Simple database initialization script for authentication
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import engine
from app.models.user import Base

async def init_database():
    """Initialize the database with basic tables"""
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False
    return True

if __name__ == "__main__":
    asyncio.run(init_database())