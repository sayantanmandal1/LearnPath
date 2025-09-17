#!/usr/bin/env python3
"""
Test Redis connectivity
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.redis import redis_manager

async def test_redis():
    """Test Redis connection"""
    try:
        await redis_manager.connect()
        print("✅ Redis connected successfully!")
        
        # Test basic operations
        await redis_manager.set("test_key", "test_value", ttl=60)
        value = await redis_manager.get("test_key")
        print(f"✅ Redis set/get test: {value}")
        
        await redis_manager.delete("test_key")
        print("✅ Redis delete test successful")
        
        await redis_manager.disconnect()
        print("✅ Redis disconnected successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_redis())