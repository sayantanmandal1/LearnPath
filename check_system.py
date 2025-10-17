#!/usr/bin/env python3
"""
System status checker for AI Career Recommender
Verifies all components are working correctly
"""

import requests
import json
import sys
from datetime import datetime

def check_service(name, url, expected_status=200):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == expected_status:
            print(f"✅ {name}: OK ({response.status_code})")
            return True
        else:
            print(f"❌ {name}: Failed ({response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: Connection failed - {e}")
        return False

def check_api_endpoint(name, url, expected_keys=None):
    """Check API endpoint and validate response structure"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in data]
                if missing_keys:
                    print(f"⚠️  {name}: Missing keys {missing_keys}")
                    return False
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: Failed ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return False

def check_database_data():
    """Check if database has been initialized with data"""
    try:
        # Check skills count
        response = requests.get("http://localhost:8000/api/v1/comprehensive-api/skills?limit=1", timeout=10)
        if response.status_code == 200:
            print("✅ Database: Skills data available")
        else:
            print("⚠️  Database: Skills endpoint not accessible")
            
        # Try to get job postings (might need authentication)
        response = requests.get("http://localhost:8000/api/v1/comprehensive-api/jobs?limit=1", timeout=10)
        if response.status_code in [200, 401]:  # 401 is OK, means endpoint exists but needs auth
            print("✅ Database: Jobs endpoint available")
        else:
            print("⚠️  Database: Jobs endpoint not accessible")
            
    except Exception as e:
        print(f"⚠️  Database: Could not verify data - {e}")

def main():
    print("🔍 AI Career Recommender - System Status Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_good = True
    
    # Core services
    print("🏥 Core Services:")
    all_good &= check_service("Backend Health", "http://localhost:8000/health")
    all_good &= check_api_endpoint("API Documentation", "http://localhost:8000/api/v1/openapi.json", ["openapi", "info"])
    
    print()
    
    # API endpoints
    print("🔌 API Endpoints:")
    all_good &= check_service("API Docs", "http://localhost:8000/api/v1/docs")
    
    print()
    
    # Database
    print("💾 Database:")
    check_database_data()
    
    print()
    
    # Authentication test
    print("🔐 Authentication:")
    try:
        login_data = {
            "email": "demo@aicareer.com",
            "password": "secret"
        }
        response = requests.post(
            "http://localhost:8000/api/v1/auth/login",
            json=login_data,
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Demo Login: OK")
        else:
            print(f"⚠️  Demo Login: Failed ({response.status_code})")
    except Exception as e:
        print(f"⚠️  Demo Login: Error - {e}")
    
    print()
    
    # Summary
    print("📊 Summary:")
    if all_good:
        print("✅ All core services are operational!")
        print()
        print("🎯 Ready to use:")
        print("   • Frontend: http://localhost:3000 (run 'npm run dev' in frontend/)")
        print("   • API Docs: http://localhost:8000/api/v1/docs")
        print("   • Demo Login: demo@aicareer.com / secret")
    else:
        print("⚠️  Some services may have issues. Check the logs above.")
        print("   • Run 'docker-compose logs backend' for detailed logs")
        print("   • Ensure all services are running with 'docker-compose ps'")
    
    print()
    print("🚀 System check completed!")

if __name__ == "__main__":
    main()