"""
Data encryption utilities for sensitive user information
"""
import base64
import os
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""
    
    def __init__(self):
        self._fernet = self._initialize_fernet()
    
    def _initialize_fernet(self) -> Fernet:
        """Initialize Fernet encryption with key derived from secret"""
        # Use the JWT secret key as base for encryption key
        password = settings.JWT_SECRET_KEY.encode()
        salt = b'ai_career_salt_2024'  # Static salt for consistency
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self._fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
        
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise ValueError("Failed to encrypt data")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self._fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise ValueError("Failed to decrypt data")
    
    def encrypt_dict(self, data: dict, sensitive_fields: list[str]) -> dict:
        """Encrypt specific fields in a dictionary"""
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        
        return encrypted_data
    
    def decrypt_dict(self, data: dict, sensitive_fields: list[str]) -> dict:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field] is not None:
                try:
                    decrypted_data[field] = self.decrypt(decrypted_data[field])
                except ValueError:
                    # Field might not be encrypted, leave as is
                    logger.warning(f"Failed to decrypt field {field}, leaving as is")
        
        return decrypted_data


class PIIEncryption:
    """Specialized encryption for Personally Identifiable Information"""
    
    SENSITIVE_FIELDS = [
        'email',
        'phone_number',
        'full_name',
        'address',
        'linkedin_url',
        'github_username',
        'leetcode_id',
        'resume_content',
        'personal_notes'
    ]
    
    def __init__(self):
        self.encryption_service = EncryptionService()
    
    def encrypt_user_data(self, user_data: dict) -> dict:
        """Encrypt PII fields in user data"""
        return self.encryption_service.encrypt_dict(user_data, self.SENSITIVE_FIELDS)
    
    def decrypt_user_data(self, encrypted_data: dict) -> dict:
        """Decrypt PII fields in user data"""
        return self.encryption_service.decrypt_dict(encrypted_data, self.SENSITIVE_FIELDS)
    
    def hash_for_search(self, value: str) -> str:
        """Create searchable hash of sensitive data"""
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()


# Global instances
encryption_service = EncryptionService()
pii_encryption = PIIEncryption()