#!/usr/bin/env python3
"""
Environment setup script for AI Career Recommender
Creates .env files from templates with secure defaults
"""

import os
import shutil
import secrets
import string
from pathlib import Path

def generate_secret_key(length=32):
    """Generate a secure random secret key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def setup_backend_env():
    """Setup backend .env file from template"""
    backend_dir = Path("backend")
    env_file = backend_dir / ".env"
    template_file = backend_dir / ".env.template"
    
    if env_file.exists():
        response = input(f"Backend .env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping backend .env setup")
            return
    
    if not template_file.exists():
        print("❌ Backend .env.template not found!")
        return
    
    # Read template
    with open(template_file, 'r') as f:
        content = f.read()
    
    # Replace placeholders with secure values
    replacements = {
        "your-secret-key-here-change-in-production-32-chars": generate_secret_key(32),
        "your-jwt-secret-key-here-change-in-production-32-chars": generate_secret_key(32),
        "your-gemini-api-key-here": "AIzaSyDgr4rW2ts3Sj012r5Sqk52EWudTC5WNNI"  # Demo key
    }
    
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Backend .env file created with secure random keys")

def setup_frontend_env():
    """Setup frontend .env file from template"""
    frontend_dir = Path("frontend")
    env_file = frontend_dir / ".env"
    template_file = frontend_dir / ".env.template"
    
    if env_file.exists():
        response = input(f"Frontend .env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping frontend .env setup")
            return
    
    if not template_file.exists():
        print("❌ Frontend .env.template not found!")
        return
    
    # Read template
    with open(template_file, 'r') as f:
        content = f.read()
    
    # Replace placeholders with demo values
    replacements = {
        "your-supabase-url-here": "https://bmhvwzqadllsyncnyhyw.supabase.co",
        "your-supabase-anon-key-here": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJtaHZ3enFhZGxsc3luY255aHl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc4NDE0ODAsImV4cCI6MjA3MzQxNzQ4MH0.DYSVsBjkiYfLEv1dJfECT2RF8XS2hnSwDL9iIW799co"
    }
    
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Frontend .env file created")

def main():
    print("🔧 AI Career Recommender - Environment Setup")
    print("=" * 50)
    print()
    
    print("This script will create .env files from templates with secure random keys.")
    print("⚠️  Make sure to update API keys and other sensitive values before production!")
    print()
    
    # Setup backend environment
    print("📁 Setting up backend environment...")
    setup_backend_env()
    print()
    
    # Setup frontend environment
    print("📁 Setting up frontend environment...")
    setup_frontend_env()
    print()
    
    print("✅ Environment setup completed!")
    print()
    print("🔒 Security Notes:")
    print("   • Secret keys have been randomly generated")
    print("   • Update GEMINI_API_KEY with your actual API key")
    print("   • Update Supabase credentials if using different instance")
    print("   • Never commit .env files to version control")
    print()
    print("🚀 Next steps:")
    print("   1. Review and update .env files as needed")
    print("   2. Run 'docker-compose up -d' to start services")
    print("   3. Run 'python check_system.py' to verify setup")

if __name__ == "__main__":
    main()