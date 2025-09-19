#!/usr/bin/env python3
"""
Backend API Testing Script
Tests all major endpoints with curl commands to ensure everything works properly
"""
import subprocess
import json
import time
import sys

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def run_curl(url, method="GET", data=None, headers=None, auth_token=None):
    """Run curl command and return response"""
    cmd = ["curl", "-s", "-X", method]
    
    if headers:
        for key, value in headers.items():
            cmd.extend(["-H", f"{key}: {value}"])
    
    if auth_token:
        cmd.extend(["-H", f"Authorization: Bearer {auth_token}"])
    
    if data:
        cmd.extend(["-H", "Content-Type: application/json"])
        cmd.extend(["-d", json.dumps(data)])
    
    cmd.append(url)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return None, "Request timed out", 1

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    stdout, stderr, code = run_curl(f"{BASE_URL}/health")
    
    if code == 0:
        try:
            response = json.loads(stdout)
            if response.get("status") == "healthy":
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response}")
                return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return False
    else:
        print(f"❌ Health check failed: {stderr}")
        return False

def test_user_registration():
    """Test user registration"""
    print("\n🔍 Testing user registration...")
    
    user_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/auth/register",
        method="POST",
        data=user_data
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            if "access_token" in response:
                print("✅ User registration successful")
                return response["access_token"]
            else:
                print(f"❌ Registration failed: {response}")
                return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return None
    else:
        print(f"❌ Registration request failed: {stderr}")
        return None

def test_user_login():
    """Test user login"""
    print("\n🔍 Testing user login...")
    
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/auth/login",
        method="POST",
        data=login_data
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            if "access_token" in response:
                print("✅ User login successful")
                return response["access_token"]
            else:
                print(f"❌ Login failed: {response}")
                return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return None
    else:
        print(f"❌ Login request failed: {stderr}")
        return None

def test_profile_creation(auth_token):
    """Test profile creation"""
    print("\n🔍 Testing profile creation...")
    
    profile_data = {
        "name": "Test User",
        "email": "test@example.com",
        "skills": ["Python", "JavaScript", "React"],
        "experience_years": 3,
        "current_role": "Software Developer",
        "education": "Bachelor's in Computer Science",
        "location": "Bangalore, India"
    }
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/profiles",
        method="POST",
        data=profile_data,
        auth_token=auth_token
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            if "id" in response:
                print("✅ Profile creation successful")
                return response["id"]
            else:
                print(f"❌ Profile creation failed: {response}")
                return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return None
    else:
        print(f"❌ Profile creation request failed: {stderr}")
        return None

def test_dashboard_data(auth_token):
    """Test dashboard data endpoint"""
    print("\n🔍 Testing dashboard data...")
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/dashboard/summary",
        auth_token=auth_token
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            print("✅ Dashboard data retrieved successfully")
            print(f"   Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            return True
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return False
    else:
        print(f"❌ Dashboard data request failed: {stderr}")
        return False

def test_job_recommendations(auth_token):
    """Test job recommendations endpoint"""
    print("\n🔍 Testing job recommendations...")
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/dashboard/job-recommendations?limit=5",
        auth_token=auth_token
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            print("✅ Job recommendations retrieved successfully")
            if isinstance(response, dict) and "job_matches" in response:
                print(f"   Found {len(response['job_matches'])} job matches")
            return True
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return False
    else:
        print(f"❌ Job recommendations request failed: {stderr}")
        return False

def test_market_insights(auth_token):
    """Test market insights endpoint"""
    print("\n🔍 Testing market insights...")
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/market-insights?location=India",
        auth_token=auth_token
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            print("✅ Market insights retrieved successfully")
            if isinstance(response, dict):
                print(f"   Response keys: {list(response.keys())}")
            return True
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return False
    else:
        print(f"❌ Market insights request failed: {stderr}")
        return False

def test_learning_paths(auth_token, user_id):
    """Test learning paths endpoint"""
    print("\n🔍 Testing learning paths...")
    
    stdout, stderr, code = run_curl(
        f"{API_BASE}/learning-paths/{user_id}",
        auth_token=auth_token
    )
    
    if code == 0:
        try:
            response = json.loads(stdout)
            print("✅ Learning paths retrieved successfully")
            if isinstance(response, list):
                print(f"   Found {len(response)} learning paths")
            elif isinstance(response, dict) and "learning_paths" in response:
                print(f"   Found {len(response['learning_paths'])} learning paths")
            return True
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {stdout}")
            return False
    else:
        print(f"❌ Learning paths request failed: {stderr}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Backend API Tests")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("\n❌ Health check failed. Make sure the backend server is running.")
        sys.exit(1)
    
    # Test user registration (or login if user exists)
    auth_token = test_user_registration()
    if not auth_token:
        # Try login instead
        auth_token = test_user_login()
        if not auth_token:
            print("\n❌ Authentication failed. Cannot continue with tests.")
            sys.exit(1)
    
    # Test profile creation
    user_id = test_profile_creation(auth_token)
    
    # Test dashboard endpoints
    test_dashboard_data(auth_token)
    test_job_recommendations(auth_token)
    test_market_insights(auth_token)
    
    if user_id:
        test_learning_paths(auth_token, user_id)
    
    print("\n" + "=" * 50)
    print("🎉 Backend API tests completed!")
    print("\nTo run manual curl tests, use the following commands:")
    print(f"export AUTH_TOKEN='{auth_token}'")
    print(f"curl -H 'Authorization: Bearer $AUTH_TOKEN' {API_BASE}/dashboard/summary")
    print(f"curl -H 'Authorization: Bearer $AUTH_TOKEN' {API_BASE}/dashboard/job-recommendations")
    print(f"curl -H 'Authorization: Bearer $AUTH_TOKEN' {API_BASE}/market-insights")

if __name__ == "__main__":
    main()