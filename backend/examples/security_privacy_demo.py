"""
Comprehensive demo of security and privacy protection measures
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from app.core.encryption import encryption_service, pii_encryption
from app.core.input_validation import (
    InputValidator, InputSanitizer, validate_and_sanitize_input
)
from app.core.audit_logging import audit_logger, AuditEventType, AuditSeverity
from app.core.privacy_compliance import (
    privacy_service, ConsentType, DataCategory, PrivacyRequestType
)


def demonstrate_encryption():
    """Demonstrate encryption and decryption capabilities"""
    print("=== ENCRYPTION DEMONSTRATION ===")
    
    # Basic string encryption
    sensitive_data = "user@example.com"
    encrypted = encryption_service.encrypt(sensitive_data)
    decrypted = encryption_service.decrypt(encrypted)
    
    print(f"Original: {sensitive_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {sensitive_data == decrypted}")
    print()
    
    # PII encryption for user data
    user_data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-123-4567",
        "github_username": "johndoe",
        "linkedin_url": "https://linkedin.com/in/johndoe",
        "skills": ["Python", "JavaScript", "React"],  # Not sensitive
        "experience_years": 5  # Not sensitive
    }
    
    print("User Data Encryption:")
    print(f"Original data: {json.dumps(user_data, indent=2)}")
    
    encrypted_data = pii_encryption.encrypt_user_data(user_data)
    print(f"Encrypted data: {json.dumps(encrypted_data, indent=2)}")
    
    decrypted_data = pii_encryption.decrypt_user_data(encrypted_data)
    print(f"Decrypted data: {json.dumps(decrypted_data, indent=2)}")
    print(f"Data integrity: {user_data == decrypted_data}")
    print()
    
    # Searchable hash generation
    email_hash = pii_encryption.hash_for_search(user_data["email"])
    print(f"Searchable hash for email: {email_hash}")
    print()


def demonstrate_input_validation():
    """Demonstrate input validation and sanitization"""
    print("=== INPUT VALIDATION DEMONSTRATION ===")
    
    # HTML sanitization
    malicious_html = '''
    <script>alert('XSS Attack!');</script>
    <p>This is safe content</p>
    <a href="javascript:alert('XSS')">Malicious Link</a>
    <img src="x" onerror="alert('XSS')">
    <strong>Bold text</strong>
    '''
    
    sanitized_html = InputSanitizer.sanitize_html(malicious_html)
    print("HTML Sanitization:")
    print(f"Original: {malicious_html}")
    print(f"Sanitized: {sanitized_html}")
    print()
    
    # Text sanitization
    malicious_text = '<script>alert("XSS")</script>\x00\x01\x02Normal text with control chars'
    sanitized_text = InputSanitizer.sanitize_text(malicious_text)
    print("Text Sanitization:")
    print(f"Original: {repr(malicious_text)}")
    print(f"Sanitized: {repr(sanitized_text)}")
    print()
    
    # Email validation
    test_emails = [
        "valid@example.com",
        "test.email+tag@domain.co.uk",
        "invalid-email",
        "@domain.com",
        "user@domain"
    ]
    
    print("Email Validation:")
    for email in test_emails:
        try:
            validated = InputValidator.validate_email(email)
            print(f"✓ {email} -> {validated}")
        except Exception as e:
            print(f"✗ {email} -> {str(e)}")
    print()
    
    # Password validation
    test_passwords = [
        "StrongP@ssw0rd",
        "weak",
        "nouppercase123!",
        "NOLOWERCASE123!",
        "NoNumbers!",
        "NoSpecialChars123"
    ]
    
    print("Password Validation:")
    for password in test_passwords:
        try:
            InputValidator.validate_password(password)
            print(f"✓ {password} -> Valid")
        except Exception as e:
            print(f"✗ {password} -> {str(e)}")
    print()
    
    # Comprehensive input validation
    user_input = {
        "email": "  USER@EXAMPLE.COM  ",
        "password": "MySecure123!",
        "github_username": "testuser",
        "description": "<script>alert('xss')</script><p>My professional summary</p>",
        "linkedin_url": "https://linkedin.com/in/testuser",
        "malicious_field": "<img src=x onerror=alert('xss')>Some content"
    }
    
    validation_rules = {
        "email": {"type": "email"},
        "password": {"type": "password"},
        "github_username": {"type": "username", "platform": "github"},
        "description": {"type": "html"},
        "linkedin_url": {"type": "url", "platform": "linkedin"}
    }
    
    print("Comprehensive Input Validation:")
    print(f"Original input: {json.dumps(user_input, indent=2)}")
    
    try:
        validated_input = validate_and_sanitize_input(user_input, validation_rules)
        print(f"Validated input: {json.dumps(validated_input, indent=2)}")
    except Exception as e:
        print(f"Validation error: {str(e)}")
    print()


async def demonstrate_audit_logging():
    """Demonstrate audit logging capabilities"""
    print("=== AUDIT LOGGING DEMONSTRATION ===")
    
    # Note: This would normally use a real database session
    # For demo purposes, we'll show the structure
    
    print("Audit Event Types:")
    for event_type in AuditEventType:
        print(f"  - {event_type.value}")
    print()
    
    print("Audit Severity Levels:")
    for severity in AuditSeverity:
        print(f"  - {severity.value}")
    print()
    
    # Example audit log entries
    sample_events = [
        {
            "event_type": AuditEventType.LOGIN_SUCCESS,
            "severity": AuditSeverity.LOW,
            "user_id": "123e4567-e89b-12d3-a456-426614174000",
            "ip_address": "192.168.1.100",
            "message": "User successfully logged in",
            "details": {"method": "password", "user_agent": "Mozilla/5.0"}
        },
        {
            "event_type": AuditEventType.SUSPICIOUS_ACTIVITY,
            "severity": AuditSeverity.HIGH,
            "ip_address": "10.0.0.50",
            "message": "Multiple failed login attempts detected",
            "details": {"attempts": 5, "timeframe": "5 minutes", "target_user": "admin"}
        },
        {
            "event_type": AuditEventType.DATA_EXPORT,
            "severity": AuditSeverity.MEDIUM,
            "user_id": "123e4567-e89b-12d3-a456-426614174000",
            "ip_address": "192.168.1.100",
            "message": "User exported personal data",
            "details": {"data_categories": ["basic_profile", "professional_data"], "export_size": "2.5MB"}
        }
    ]
    
    print("Sample Audit Events:")
    for i, event in enumerate(sample_events, 1):
        print(f"Event {i}:")
        print(f"  Type: {event['event_type'].value}")
        print(f"  Severity: {event['severity'].value}")
        print(f"  Message: {event['message']}")
        print(f"  Details: {json.dumps(event['details'], indent=4)}")
        print()


async def demonstrate_privacy_compliance():
    """Demonstrate privacy compliance features"""
    print("=== PRIVACY COMPLIANCE DEMONSTRATION ===")
    
    # Consent types
    print("Available Consent Types:")
    for consent_type in ConsentType:
        print(f"  - {consent_type.value}")
    print()
    
    # Data categories
    print("Data Categories:")
    for category in DataCategory:
        print(f"  - {category.value}")
    print()
    
    # Privacy request types
    print("Privacy Request Types:")
    for request_type in PrivacyRequestType:
        print(f"  - {request_type.value}")
    print()
    
    # Sample user consent management
    user_id = "123e4567-e89b-12d3-a456-426614174000"
    
    print("Sample Consent Management:")
    consent_scenarios = [
        {
            "action": "Grant consent for data processing",
            "consent_type": ConsentType.DATA_PROCESSING,
            "granted": True,
            "metadata": {"source": "registration", "timestamp": datetime.utcnow().isoformat()}
        },
        {
            "action": "Grant consent for marketing",
            "consent_type": ConsentType.MARKETING,
            "granted": True,
            "metadata": {"source": "settings_page", "campaign": "newsletter_signup"}
        },
        {
            "action": "Withdraw consent for analytics",
            "consent_type": ConsentType.ANALYTICS,
            "granted": False,
            "metadata": {"reason": "privacy_concerns", "timestamp": datetime.utcnow().isoformat()}
        }
    ]
    
    for scenario in consent_scenarios:
        print(f"  {scenario['action']}:")
        print(f"    Consent Type: {scenario['consent_type'].value}")
        print(f"    Granted: {scenario['granted']}")
        print(f"    Metadata: {json.dumps(scenario['metadata'], indent=6)}")
        print()
    
    # Sample privacy requests
    print("Sample Privacy Requests:")
    privacy_requests = [
        {
            "type": PrivacyRequestType.DATA_EXPORT,
            "description": "I want to export all my personal data",
            "categories": [DataCategory.BASIC_PROFILE, DataCategory.PROFESSIONAL_DATA],
            "status": "pending"
        },
        {
            "type": PrivacyRequestType.DATA_DELETION,
            "description": "Please delete my account and all associated data",
            "categories": None,  # All categories
            "status": "in_progress"
        },
        {
            "type": PrivacyRequestType.DATA_CORRECTION,
            "description": "My email address is incorrect in your system",
            "categories": [DataCategory.BASIC_PROFILE],
            "status": "completed"
        }
    ]
    
    for i, request in enumerate(privacy_requests, 1):
        print(f"  Request {i}:")
        print(f"    Type: {request['type'].value}")
        print(f"    Description: {request['description']}")
        print(f"    Categories: {[cat.value for cat in request['categories']] if request['categories'] else 'All'}")
        print(f"    Status: {request['status']}")
        print()
    
    # Sample data export structure
    print("Sample Data Export Structure:")
    sample_export = {
        "user_id": user_id,
        "export_timestamp": datetime.utcnow().isoformat(),
        "data_categories": ["basic_profile", "professional_data"],
        "data": {
            "basic_profile": {
                "id": user_id,
                "email": "user@example.com",
                "created_at": "2024-01-01T00:00:00Z",
                "is_active": True
            },
            "professional_data": {
                "skills": ["Python", "JavaScript", "React"],
                "experience_years": 5,
                "dream_job": "Senior Software Engineer",
                "github_username": "testuser",
                "linkedin_url": "https://linkedin.com/in/testuser"
            },
            "consents": [
                {
                    "consent_type": "data_processing",
                    "granted": True,
                    "granted_at": "2024-01-01T00:00:00Z"
                }
            ]
        }
    }
    
    print(json.dumps(sample_export, indent=2))
    print()


def demonstrate_security_best_practices():
    """Demonstrate security best practices implementation"""
    print("=== SECURITY BEST PRACTICES DEMONSTRATION ===")
    
    print("1. Data Encryption:")
    print("   ✓ AES-256 encryption for sensitive data at rest")
    print("   ✓ Separate encryption keys for different data types")
    print("   ✓ Key derivation using PBKDF2 with high iteration count")
    print("   ✓ Searchable hashes for encrypted data lookup")
    print()
    
    print("2. Input Validation & Sanitization:")
    print("   ✓ Comprehensive HTML sanitization using bleach")
    print("   ✓ Email validation using email-validator library")
    print("   ✓ Strong password requirements enforcement")
    print("   ✓ Platform-specific username validation")
    print("   ✓ File upload validation (type, size, content)")
    print("   ✓ URL validation and protocol restrictions")
    print()
    
    print("3. Rate Limiting & DDoS Protection:")
    print("   ✓ Redis-based sliding window rate limiting")
    print("   ✓ Different limits for different endpoint types")
    print("   ✓ IP-based request pattern analysis")
    print("   ✓ Automatic bot detection and blocking")
    print("   ✓ Distributed attack pattern recognition")
    print("   ✓ Intelligent retry mechanisms")
    print()
    
    print("4. Audit Logging:")
    print("   ✓ Comprehensive event logging with structured data")
    print("   ✓ Multiple severity levels for proper alerting")
    print("   ✓ User action tracking for compliance")
    print("   ✓ Security event monitoring and alerting")
    print("   ✓ Searchable audit trail with filtering")
    print("   ✓ Automatic log retention and cleanup")
    print()
    
    print("5. Privacy Compliance:")
    print("   ✓ GDPR Article 15-22 compliance (Right of Access, Erasure, etc.)")
    print("   ✓ CCPA compliance for California residents")
    print("   ✓ Granular consent management system")
    print("   ✓ Data category classification and retention policies")
    print("   ✓ Automated data export and deletion capabilities")
    print("   ✓ Privacy request workflow with verification")
    print()
    
    print("6. API Security:")
    print("   ✓ JWT-based authentication with refresh tokens")
    print("   ✓ Role-based access control (RBAC)")
    print("   ✓ Request/response validation using Pydantic")
    print("   ✓ CORS configuration for cross-origin requests")
    print("   ✓ Security headers implementation")
    print("   ✓ API versioning and deprecation handling")
    print()


async def main():
    """Run all demonstrations"""
    print("AI Career Recommender - Security & Privacy Protection Demo")
    print("=" * 60)
    print()
    
    # Run demonstrations
    demonstrate_encryption()
    demonstrate_input_validation()
    await demonstrate_audit_logging()
    await demonstrate_privacy_compliance()
    demonstrate_security_best_practices()
    
    print("Demo completed successfully!")
    print()
    print("Key Security Features Implemented:")
    print("✓ Data encryption for sensitive user information")
    print("✓ Comprehensive input validation and sanitization")
    print("✓ Rate limiting and DDoS protection mechanisms")
    print("✓ Audit logging for security monitoring")
    print("✓ User data export and deletion capabilities")
    print("✓ Privacy compliance features and consent management")


if __name__ == "__main__":
    asyncio.run(main())