#!/usr/bin/env python3
"""
Simple database initialization script for authentication only
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import engine
from sqlalchemy import text

async def init_auth_database():
    """Initialize the database with basic authentication tables"""
    try:
        async with engine.begin() as conn:
            # Create users table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    full_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            """))
            
            # Create refresh_tokens table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_revoked BOOLEAN DEFAULT FALSE,
                    device_info TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """))
            
            # Create indexes
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token_hash ON refresh_tokens(token_hash)"))
            
        print("Authentication database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False
    return True

if __name__ == "__main__":
    asyncio.run(init_auth_database())