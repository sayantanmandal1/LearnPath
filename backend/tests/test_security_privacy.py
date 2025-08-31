"""
Tests for security and privacy protection measures
"""
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.encryption import encryption_service, pii_encryption
from app.core.input_validation import (
    InputValidator, InputSanitizer, ValidationError, validate_and_sanitize_input
)
from app.core.rate_limiting import RateLimiter, DDoSProtection
from app.core.audit_logging import audit_logger, AuditEventType, AuditSeverity
from app.core.privacy_compliance import (
    privacy_service, ConsentType, DataCategory, PrivacyRequestType
)
from app.models.user import User


class TestEncryption:
    """Test encryption and decryption functionality"""
    
    def test_encrypt_decrypt_string(self):
        """Test basic string encryption and decryption"""
        original_data = "sensitive user information"
        
        # Encrypt
        encrypted_data = encryption_service.encrypt(original_data)
        assert encrypted_data != original_data
        assert len(encrypted_data) > len(original_data)
        
        # Decrypt
        decrypted_data = encryption_service.decrypt(encrypted_data)
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_bytes(self):
        """Test bytes encryption and decryption"""
        original_data = b"sensitive binary data"
        
        # Encrypt
        encrypted_data = encryption_service.encrypt(original_data)
        assert encrypted_data != original_data.decode()
        
        # Decrypt
        decrypted_data = encryption_service.decrypt(encrypted_data)
        assert decrypted_data == original_data.decode()
    
    def test_encrypt_dict_fields(self):
        """Test dictionary field encryption"""
        original_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "public_info": "This is public",
            "phone": "123-456-7890"
        }
        
        sensitive_fields = ["email", "phone"]
        
        # Encrypt sensitive fields
        encrypted_data = encryption_service.encrypt_dict(original_data, sensitive_fields)
        
        assert encrypted_data["name"] == original_data["name"]  # Not encrypted
        assert encrypted_data["public_info"] == original_data["public_info"]  # Not encrypted
        assert encrypted_data["email"] != original_data["email"]  # Encrypted
        assert encrypted_data["phone"] != original_data["phone"]  # Encrypted
        
        # Decrypt sensitive fields
        decrypted_data = encryption_service.decrypt_dict(encrypted_data, sensitive_fields)
        
        assert decrypted_data == original_data
    
    def test_pii_encryption(self):
        """Test PII-specific encryption"""
        user_data = {
            "email": "user@example.com",
            "github_username": "testuser",
            "linkedin_url": "https://linkedin.com/in/testuser",
            "public_skills": ["Python", "JavaScript"]
        }
        
        # Encrypt PII
        encrypted_data = pii_encryption.encrypt_user_data(user_data)
        
        # Check that sensitive fields are encrypted
        assert encrypted_data["email"] != user_data["email"]
        assert encrypted_data["github_username"] != user_data["github_username"]
        assert encrypted_data["linkedin_url"] != user_data["linkedin_url"]
        assert encrypted_data["public_skills"] == user_data["public_skills"]  # Not in sensitive fields
        
        # Decrypt PII
        decrypted_data = pii_encryption.decrypt_user_data(encrypted_data)
        
        assert decrypted_data["email"] == user_data["email"]
        assert decrypted_data["github_username"] == user_data["github_username"]
        assert decrypted_data["linkedin_url"] == user_data["linkedin_url"]
    
    def test_hash_for_search(self):
        """Test searchable hash generation"""
        value = "test@example.com"
        
        hash1 = pii_encryption.hash_for_search(value)
        hash2 = pii_encryption.hash_for_search(value)
        hash3 = pii_encryption.hash_for_search("different@example.com")
        
        # Same value should produce same hash
        assert hash1 == hash2
        
        # Different values should produce different hashes
        assert hash1 != hash3
        
        # Hash should be deterministic and not reveal original value
        assert len(hash1) == 64  # SHA256 hex length
        assert value not in hash1


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_sanitize_html(self):
        """Test HTML sanitization"""
        malicious_html = '<script>alert("xss")</script><p>Safe content</p><a href="javascript:alert()">Link</a>'
        
        sanitized = InputSanitizer.sanitize_html(malicious_html)
        
        assert '<script>' not in sanitized
        assert 'alert(' not in sanitized
        assert '<p>Safe content</p>' in sanitized
        assert 'javascript:' not in sanitized
    
    def test_sanitize_text(self):
        """Test text sanitization"""
        malicious_text = '<script>alert("xss")</script>\x00\x01Normal text'
        
        sanitized = InputSanitizer.sanitize_text(malicious_text)
        
        assert '&lt;script&gt;' in sanitized  # HTML escaped
        assert '\x00' not in sanitized  # Control characters removed
        assert '\x01' not in sanitized
        assert 'Normal text' in sanitized
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        dangerous_filename = '../../../etc/passwd<>:"|?*\x00.txt'
        
        sanitized = InputSanitizer.sanitize_filename(dangerous_filename)
        
        assert '../' not in sanitized
        assert '<' not in sanitized
        assert '>' not in sanitized
        assert ':' not in sanitized
        assert '"' not in sanitized
        assert '|' not in sanitized
        assert '?' not in sanitized
        assert '*' not in sanitized
        assert '\x00' not in sanitized
    
    def test_sanitize_url(self):
        """Test URL sanitization"""
        valid_url = "https://example.com/path"
        invalid_url = "javascript:alert('xss')"
        
        # Valid URL should pass
        sanitized_valid = InputSanitizer.sanitize_url(valid_url)
        assert sanitized_valid == valid_url
        
        # Invalid URL should raise exception
        with pytest.raises(ValidationError):
            InputSanitizer.sanitize_url(invalid_url)
    
    def test_validate_email(self):
        """Test email validation"""
        valid_emails = [
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "user123@test-domain.com"
        ]
        
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user space@domain.com"
        ]
        
        for email in valid_emails:
            validated = InputValidator.validate_email(email)
            assert validated == email.lower()
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                InputValidator.validate_email(email)
    
    def test_validate_password(self):
        """Test password validation"""
        valid_passwords = [
            "StrongP@ssw0rd",
            "MySecure123!",
            "C0mplex&Password"
        ]
        
        invalid_passwords = [
            "weak",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoNumbers!",  # No digits
            "NoSpecialChars123",  # No special characters
            "AAA111!!!",  # Consecutive identical characters
        ]
        
        for password in valid_passwords:
            validated = InputValidator.validate_password(password)
            assert validated == password
        
        for password in invalid_passwords:
            with pytest.raises(ValidationError):
                InputValidator.validate_password(password)
    
    def test_validate_username(self):
        """Test username validation"""
        # GitHub usernames
        valid_github = ["user123", "test-user", "a", "a" * 39]
        invalid_github = ["-user", "user-", "a" * 40, "user@name"]
        
        for username in valid_github:
            validated = InputValidator.validate_username(username, "github")
            assert validated == username
        
        for username in invalid_github:
            with pytest.raises(ValidationError):
                InputValidator.validate_username(username, "github")
        
        # LeetCode usernames
        valid_leetcode = ["user123", "test_user", "a", "a" * 20]
        invalid_leetcode = ["a" * 21, "user@name", "user space"]
        
        for username in valid_leetcode:
            validated = InputValidator.validate_username(username, "leetcode")
            assert validated == username
        
        for username in invalid_leetcode:
            with pytest.raises(ValidationError):
                InputValidator.validate_username(username, "leetcode")
    
    def test_validate_file_upload(self):
        """Test file upload validation"""
        # Valid files
        valid_files = [
            ("resume.pdf", "application/pdf", 1024 * 1024),  # 1MB PDF
            ("document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 2 * 1024 * 1024),  # 2MB DOCX
        ]
        
        for filename, content_type, file_size in valid_files:
            # Should not raise exception
            InputValidator.validate_file_upload(filename, content_type, file_size)
        
        # Invalid files
        invalid_files = [
            ("malware.exe", "application/octet-stream", 1024),  # Invalid extension
            ("resume.pdf", "text/plain", 1024),  # Mismatched content type
            ("huge.pdf", "application/pdf", 20 * 1024 * 1024),  # Too large (20MB)
        ]
        
        for filename, content_type, file_size in invalid_files:
            with pytest.raises(ValidationError):
                InputValidator.validate_file_upload(filename, content_type, file_size)
    
    def test_validate_and_sanitize_input(self):
        """Test comprehensive input validation and sanitization"""
        input_data = {
            "email": "  USER@EXAMPLE.COM  ",
            "password": "StrongP@ssw0rd",
            "github_username": "testuser",
            "description": "<script>alert('xss')</script>Safe content",
            "linkedin_url": "https://linkedin.com/in/testuser"
        }
        
        validation_rules = {
            "email": {"type": "email"},
            "password": {"type": "password"},
            "github_username": {"type": "username", "platform": "github"},
            "description": {"type": "html"},
            "linkedin_url": {"type": "url", "platform": "linkedin"}
        }
        
        validated_data = validate_and_sanitize_input(input_data, validation_rules)
        
        assert validated_data["email"] == "user@example.com"  # Normalized
        assert validated_data["password"] == "StrongP@ssw0rd"
        assert validated_data["github_username"] == "testuser"
        assert "<script>" not in validated_data["description"]  # Sanitized
        assert "Safe content" in validated_data["description"]
        assert validated_data["linkedin_url"] == input_data["linkedin_url"]


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis_mock = Mock()
        redis_mock.pipeline.return_value = redis_mock
        redis_mock.zremrangebyscore.return_value = None
        redis_mock.zcard.return_value = 0
        redis_mock.zadd.return_value = None
        redis_mock.expire.return_value = None
        redis_mock.execute.return_value = [None, 0, None, None]
        return redis_mock
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self, mock_redis):
        """Test that rate limiter allows requests within limit"""
        rate_limiter = RateLimiter(mock_redis)
        
        # Mock Redis responses for allowed request
        mock_redis.execute.return_value = [None, 5, None, None]  # 5 current requests
        
        is_allowed, retry_after = await rate_limiter.is_allowed("test_key", 10, 60)
        
        assert is_allowed is True
        assert retry_after == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self, mock_redis):
        """Test that rate limiter blocks requests over limit"""
        rate_limiter = RateLimiter(mock_redis)
        
        # Mock Redis responses for blocked request
        mock_redis.execute.return_value = [None, 15, None, None]  # 15 current requests (over limit of 10)
        mock_redis.zrange.return_value = [(b'123456789', 123456789.0)]
        
        is_allowed, retry_after = await rate_limiter.is_allowed("test_key", 10, 60)
        
        assert is_allowed is False
        assert retry_after > 0
    
    @pytest.mark.asyncio
    async def test_ddos_protection_analysis(self, mock_redis):
        """Test DDoS protection request analysis"""
        ddos_protection = DDoSProtection(mock_redis)
        
        # Mock Redis responses
        mock_redis.get.return_value = None
        mock_redis.incr.return_value = None
        mock_redis.expire.return_value = None
        
        analysis = await ddos_protection.analyze_request_pattern(
            "192.168.1.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "/api/v1/profiles"
        )
        
        assert "is_suspicious" in analysis
        assert "risk_score" in analysis
        assert "reasons" in analysis
        assert isinstance(analysis["is_suspicious"], bool)
        assert isinstance(analysis["risk_score"], int)
        assert isinstance(analysis["reasons"], list)
    
    @pytest.mark.asyncio
    async def test_ddos_protection_bot_detection(self, mock_redis):
        """Test DDoS protection bot detection"""
        ddos_protection = DDoSProtection(mock_redis)
        
        # Mock Redis responses
        mock_redis.get.return_value = None
        mock_redis.incr.return_value = None
        mock_redis.expire.return_value = None
        
        analysis = await ddos_protection.analyze_request_pattern(
            "192.168.1.1",
            "python-requests/2.28.1",  # Bot-like user agent
            "/api/v1/profiles"
        )
        
        assert analysis["risk_score"] >= 20  # Should add points for bot-like user agent
        assert any("bot" in reason.lower() for reason in analysis["reasons"])


class TestAuditLogging:
    """Test audit logging functionality"""
    
    @pytest.mark.asyncio
    async def test_log_authentication_event(self, db_session: AsyncSession):
        """Test logging authentication events"""
        user_id = str(uuid.uuid4())
        
        await audit_logger.log_authentication_event(
            db=db_session,
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id=user_id,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True,
            details={"method": "password"}
        )
        
        # Verify log was created
        logs = await audit_logger.search_audit_logs(
            db=db_session,
            event_types=[AuditEventType.LOGIN_SUCCESS],
            user_id=user_id,
            limit=1
        )
        
        assert len(logs) == 1
        assert logs[0].event_type == AuditEventType.LOGIN_SUCCESS.value
        assert logs[0].user_id == uuid.UUID(user_id)
        assert logs[0].ip_address == "192.168.1.1"
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, db_session: AsyncSession):
        """Test logging security events"""
        await audit_logger.log_security_event(
            db=db_session,
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            message="Multiple failed login attempts",
            ip_address="192.168.1.100",
            severity=AuditSeverity.HIGH,
            details={"attempts": 5, "timeframe": "5 minutes"}
        )
        
        # Verify log was created
        logs = await audit_logger.search_audit_logs(
            db=db_session,
            event_types=[AuditEventType.SUSPICIOUS_ACTIVITY],
            severity=AuditSeverity.HIGH,
            limit=1
        )
        
        assert len(logs) == 1
        assert logs[0].severity == AuditSeverity.HIGH.value
        assert logs[0].details["attempts"] == 5
    
    @pytest.mark.asyncio
    async def test_search_audit_logs_with_filters(self, db_session: AsyncSession):
        """Test searching audit logs with various filters"""
        user_id = str(uuid.uuid4())
        
        # Create multiple log entries
        await audit_logger.log_event(
            db=db_session,
            event_type=AuditEventType.LOGIN_SUCCESS,
            message="User logged in",
            user_id=user_id,
            ip_address="192.168.1.1",
            severity=AuditSeverity.LOW
        )
        
        await audit_logger.log_event(
            db=db_session,
            event_type=AuditEventType.PROFILE_UPDATED,
            message="Profile updated",
            user_id=user_id,
            ip_address="192.168.1.1",
            severity=AuditSeverity.LOW
        )
        
        # Search by user ID
        user_logs = await audit_logger.search_audit_logs(
            db=db_session,
            user_id=user_id,
            limit=10
        )
        
        assert len(user_logs) == 2
        assert all(log.user_id == uuid.UUID(user_id) for log in user_logs)
        
        # Search by event type
        login_logs = await audit_logger.search_audit_logs(
            db=db_session,
            event_types=[AuditEventType.LOGIN_SUCCESS],
            limit=10
        )
        
        assert len(login_logs) >= 1
        assert all(log.event_type == AuditEventType.LOGIN_SUCCESS.value for log in login_logs)


class TestPrivacyCompliance:
    """Test privacy compliance functionality"""
    
    @pytest.mark.asyncio
    async def test_grant_consent(self, db_session: AsyncSession):
        """Test granting user consent"""
        user_id = str(uuid.uuid4())
        
        consent = await privacy_service.grant_consent(
            db=db_session,
            user_id=user_id,
            consent_type=ConsentType.DATA_PROCESSING,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            metadata={"source": "registration"}
        )
        
        assert consent.user_id == uuid.UUID(user_id)
        assert consent.consent_type == ConsentType.DATA_PROCESSING.value
        assert consent.granted is True
        assert consent.granted_at is not None
        assert consent.metadata["source"] == "registration"
    
    @pytest.mark.asyncio
    async def test_withdraw_consent(self, db_session: AsyncSession):
        """Test withdrawing user consent"""
        user_id = str(uuid.uuid4())
        
        # First grant consent
        await privacy_service.grant_consent(
            db=db_session,
            user_id=user_id,
            consent_type=ConsentType.MARKETING,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Then withdraw it
        consent = await privacy_service.withdraw_consent(
            db=db_session,
            user_id=user_id,
            consent_type=ConsentType.MARKETING,
            ip_address="192.168.1.1"
        )
        
        assert consent.granted is False
        assert consent.withdrawn_at is not None
    
    @pytest.mark.asyncio
    async def test_check_consent(self, db_session: AsyncSession):
        """Test checking user consent status"""
        user_id = str(uuid.uuid4())
        
        # Initially no consent
        has_consent = await privacy_service.check_consent(
            db=db_session,
            user_id=user_id,
            consent_type=ConsentType.ANALYTICS
        )
        assert has_consent is False
        
        # Grant consent
        await privacy_service.grant_consent(
            db=db_session,
            user_id=user_id,
            consent_type=ConsentType.ANALYTICS,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Now should have consent
        has_consent = await privacy_service.check_consent(
            db=db_session,
            user_id=user_id,
            consent_type=ConsentType.ANALYTICS
        )
        assert has_consent is True
    
    @pytest.mark.asyncio
    async def test_create_privacy_request(self, db_session: AsyncSession):
        """Test creating privacy requests"""
        user_id = str(uuid.uuid4())
        
        privacy_request = await privacy_service.create_privacy_request(
            db=db_session,
            user_id=user_id,
            request_type=PrivacyRequestType.DATA_EXPORT,
            description="I want to export all my data",
            requested_data_categories=[DataCategory.BASIC_PROFILE, DataCategory.PROFESSIONAL_DATA],
            ip_address="192.168.1.1"
        )
        
        assert privacy_request.user_id == uuid.UUID(user_id)
        assert privacy_request.request_type == PrivacyRequestType.DATA_EXPORT.value
        assert privacy_request.description == "I want to export all my data"
        assert privacy_request.verification_token is not None
        assert privacy_request.expires_at > datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_export_user_data(self, db_session: AsyncSession):
        """Test exporting user data"""
        user_id = str(uuid.uuid4())
        
        # Mock user data (in real scenario, this would come from database)
        with patch('app.core.privacy_compliance.db.get') as mock_get:
            mock_user = Mock()
            mock_user.id = uuid.UUID(user_id)
            mock_user.email = "test@example.com"
            mock_user.created_at = datetime.utcnow()
            mock_user.is_active = True
            mock_get.return_value = mock_user
            
            with patch('app.core.privacy_compliance.db.execute') as mock_execute:
                mock_execute.return_value.scalars.return_value.all.return_value = []
                
                exported_data = await privacy_service.export_user_data(
                    db=db_session,
                    user_id=user_id,
                    data_categories=[DataCategory.BASIC_PROFILE]
                )
        
        assert exported_data["user_id"] == user_id
        assert "export_timestamp" in exported_data
        assert "data" in exported_data
        assert exported_data["data_categories"] == [DataCategory.BASIC_PROFILE.value]


class TestSecurityEndpoints:
    """Test security API endpoints"""
    
    def test_get_user_security_info(self, client: TestClient, auth_headers: dict):
        """Test getting user security information"""
        response = client.get("/api/v1/security/user-security", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_id" in data
        assert "recent_logins" in data
        assert "failed_login_attempts" in data
        assert "security_score" in data
        assert "recommendations" in data
        assert isinstance(data["security_score"], int)
        assert 0 <= data["security_score"] <= 100
    
    def test_report_security_incident(self, client: TestClient, auth_headers: dict):
        """Test reporting security incidents"""
        incident_data = {
            "incident_type": "suspicious_login",
            "description": "Login attempt from unusual location"
        }
        
        response = client.post(
            "/api/v1/security/report-incident",
            json=incident_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "incident_id" in data
        assert "status" in data
        assert data["status"] == "under_review"


class TestPrivacyEndpoints:
    """Test privacy API endpoints"""
    
    def test_manage_consent(self, client: TestClient, auth_headers: dict):
        """Test managing user consent"""
        consent_data = {
            "consent_type": "data_processing",
            "granted": True,
            "metadata": {"source": "api_test"}
        }
        
        response = client.post(
            "/api/v1/privacy/consent",
            json=consent_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["consent_type"] == "data_processing"
        assert data["granted"] is True
        assert "id" in data
        assert "created_at" in data
    
    def test_get_user_consents(self, client: TestClient, auth_headers: dict):
        """Test getting user consents"""
        response = client.get("/api/v1/privacy/consent", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
    
    def test_check_specific_consent(self, client: TestClient, auth_headers: dict):
        """Test checking specific consent"""
        response = client.get(
            "/api/v1/privacy/consent/data_processing",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, bool)
    
    def test_create_privacy_request(self, client: TestClient, auth_headers: dict):
        """Test creating privacy requests"""
        request_data = {
            "request_type": "data_export",
            "description": "I want to export all my data",
            "requested_data_categories": ["basic_profile", "professional_data"]
        }
        
        response = client.post(
            "/api/v1/privacy/request",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["request_type"] == "data_export"
        assert "id" in data
        assert "expires_at" in data
        assert data["status"] == "pending"
    
    def test_get_privacy_policy(self, client: TestClient):
        """Test getting privacy policy"""
        response = client.get("/api/v1/privacy/policy")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "privacy_policy_version" in data
        assert "data_categories" in data
        assert "user_rights" in data
        assert "contact_info" in data