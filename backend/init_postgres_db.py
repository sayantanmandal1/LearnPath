#!/usr/bin/env python3
"""
PostgreSQL database initialization script for authentication
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import engine
from app.models.user import Base

async def init_postgres_database():
    """Initialize the PostgreSQL database with authentication tables"""
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        print("PostgreSQL database initialized successfully!")
        print("Tables created:")
        print("- users")
        print("- refresh_tokens")
        return True
    except Exception as e:
        print(f"Error initializing PostgreSQL database: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(init_postgres_database())