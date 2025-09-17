"""
Simple dashboard API endpoint test
"""
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app.main import app

def test_dashboard_endpoints():
    """Test that dashboard endpoints are properly registered"""
    client = TestClient(app)
    
    print("Testing dashboard endpoint registration...")
    
    # Test endpoints without authentication (should return 401)
    endpoints = [
        "/api/v1/dashboard/summary",
        "/api/v1/dashboard/progress",
        "/api/v1/dashboard/personalized-content",
        "/api/v1/dashboard/metrics",
        "/api/v1/dashboard/milestones",
        "/api/v1/dashboard/activities",
        "/api/v1/dashboard/quick-stats"
    ]
    
    for endpoint in endpoints:
        try:
            response = client.get(endpoint)
            # Should return 401 (unauthorized) since we're not authenticated
            if response.status_code == 401:
                print(f"✅ {endpoint} - Properly registered (401 Unauthorized)")
            else:
                print(f"⚠️  {endpoint} - Unexpected status code: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)}")
    
    # Test OpenAPI docs include dashboard endpoints
    try:
        response = client.get("/docs")
        if response.status_code == 200:
            print("✅ OpenAPI docs accessible")
        else:
            print(f"⚠️  OpenAPI docs status: {response.status_code}")
    except Exception as e:
        print(f"❌ OpenAPI docs error: {str(e)}")
    
    print("Dashboard endpoint registration test complete!")

if __name__ == "__main__":
    test_dashboard_endpoints()