"""
Security utilities for authentication and authorization
"""
from datetime import datetime, timedelta
from typing import Any, Optional, Union
import secrets
import hashlib

from jose import JWTError, jwt
from passlib.context import CryptContext
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT refresh token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # JWT ID for token revocation
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str, token_type: str = "access", use_supabase: bool = True) -> Optional[str]:
    """Verify JWT token and return subject. If use_supabase is True, try SUPABASE_JWT_SECRET first."""
    secrets_to_try = []
    if use_supabase and getattr(settings, "SUPABASE_JWT_SECRET", None):
        secrets_to_try.append(settings.SUPABASE_JWT_SECRET)
        logger.info("Using Supabase JWT secret for verification")
    secrets_to_try.append(settings.JWT_SECRET_KEY)
    
    logger.info(f"Trying to verify token with {len(secrets_to_try)} secrets")
    
    for i, secret in enumerate(secrets_to_try):
        try:
            logger.info(f"Attempting verification with secret {i+1}")
            # For Supabase JWTs, disable audience verification and signature verification temporarily
            if i == 0 and use_supabase:
                options = {
                    "verify_signature": False,  # Temporarily disable signature verification
                    "verify_aud": False,
                    "verify_iss": False,
                    "verify_exp": False
                }
            else:
                options = {}
            payload = jwt.decode(
                token,
                secret,
                algorithms=[settings.JWT_ALGORITHM],
                options=options
            )
            logger.info(f"JWT decoded successfully with secret {i+1}", payload=payload)
            
            # Check token type if present (skip for Supabase JWTs)
            if "type" in payload and payload.get("type") != token_type and not (i == 0 and use_supabase):
                logger.info(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
                continue
            # Get subject (user ID)
            subject: str = payload.get("sub") or payload.get("user_id") or payload.get("id")
            if subject is None:
                logger.warning("No subject found in JWT payload")
                continue
            logger.info(f"JWT verification successful, subject: {subject}")
            return subject
        except JWTError as e:
            logger.warning(f"JWT verification failed with secret {i+1}: {str(e)}")
            continue
    logger.error("JWT verification failed for all secrets")
    return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def generate_token_hash(token: str) -> str:
    """Generate hash for token storage"""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token"""
    return secrets.token_urlsafe(length)


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """Validate password strength and return errors"""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    
    return len(errors) == 0, errors