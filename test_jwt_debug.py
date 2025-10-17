"""
Simple script to test JWT verification and debug the authentication flow
"""
import requests
import json

def test_backend_api():
    """Test backend API endpoints"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test profile endpoint without auth (should get 401)
    print("\nTesting profile endpoint without auth...")
    try:
        response = requests.get(f"{base_url}/api/v1/profiles/me")
        print(f"Profile without auth: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Profile test failed: {e}")
    
    # Test with dummy token
    print("\nTesting profile endpoint with dummy token...")
    try:
        headers = {"Authorization": "Bearer dummy_token"}
        response = requests.get(f"{base_url}/api/v1/profiles/me", headers=headers)
        print(f"Profile with dummy token: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Profile with dummy token failed: {e}")

if __name__ == "__main__":
    test_backend_api()