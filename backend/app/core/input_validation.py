"""
Comprehensive input validation and sanitization
"""
import re
import html
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import bleach
import structlog
from pydantic import BaseModel, validator
from email_validator import validate_email, EmailNotValidError

logger = structlog.get_logger()


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class InputSanitizer:
    """Comprehensive input sanitization"""
    
    # Allowed HTML tags for rich text content
    ALLOWED_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
    ]
    
    ALLOWED_ATTRIBUTES = {
        '*': ['class'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'width', 'height']
    }
    
    @staticmethod
    def sanitize_html(content: str) -> str:
        """Sanitize HTML content"""
        if not content:
            return ""
        
        # Clean HTML with bleach
        cleaned = bleach.clean(
            content,
            tags=InputSanitizer.ALLOWED_TAGS,
            attributes=InputSanitizer.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        return cleaned.strip()
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize plain text input"""
        if not text:
            return ""
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return ""
        
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:250] + ('.' + ext if ext else '')
        
        return sanitized
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize and validate URL"""
        if not url:
            return ""
        
        # Basic URL validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            
            # Only allow http/https
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("Only HTTP/HTTPS URLs are allowed")
            
            return url.strip()
        
        except Exception as e:
            logger.warning("URL sanitization failed", url=url, error=str(e))
            raise ValidationError("url", "Invalid URL format")


class InputValidator:
    """Comprehensive input validation"""
    
    # Common regex patterns
    PATTERNS = {
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'phone': re.compile(r'^\+?1?-?\.?\s?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'),
        'github_username': re.compile(r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$'),
        'leetcode_username': re.compile(r'^[a-zA-Z0-9_-]{1,20}$'),
        'linkedin_url': re.compile(r'^https?://(www\.)?linkedin\.com/in/[a-zA-Z0-9-]+/?$'),
        'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
        'slug': re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$'),
    }
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address"""
        if not email:
            raise ValidationError("email", "Email is required")
        
        try:
            # Use email-validator library for comprehensive validation
            validated_email = validate_email(email)
            return validated_email.email
        except EmailNotValidError as e:
            raise ValidationError("email", str(e))
    
    @staticmethod
    def validate_password(password: str) -> str:
        """Validate password strength"""
        if not password:
            raise ValidationError("password", "Password is required")
        
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if len(password) > 128:
            errors.append("Password must be less than 128 characters")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for common patterns
        if re.search(r'(.)\1{2,}', password):
            errors.append("Password cannot contain more than 2 consecutive identical characters")
        
        if errors:
            raise ValidationError("password", "; ".join(errors))
        
        return password
    
    @staticmethod
    def validate_username(username: str, platform: str = "general") -> str:
        """Validate username for different platforms"""
        if not username:
            raise ValidationError("username", "Username is required")
        
        username = username.strip()
        
        if platform == "github":
            if not InputValidator.PATTERNS['github_username'].match(username):
                raise ValidationError("github_username", 
                    "GitHub username must be 1-39 characters, alphanumeric or hyphens, cannot start/end with hyphen")
        
        elif platform == "leetcode":
            if not InputValidator.PATTERNS['leetcode_username'].match(username):
                raise ValidationError("leetcode_username", 
                    "LeetCode username must be 1-20 characters, alphanumeric, underscores, or hyphens")
        
        else:
            # General username validation
            if len(username) < 3 or len(username) > 50:
                raise ValidationError("username", "Username must be between 3 and 50 characters")
            
            if not re.match(r'^[a-zA-Z0-9_-]+$', username):
                raise ValidationError("username", "Username can only contain letters, numbers, underscores, and hyphens")
        
        return username
    
    @staticmethod
    def validate_url(url: str, platform: str = None) -> str:
        """Validate URL with optional platform-specific checks"""
        if not url:
            return ""
        
        # Basic URL validation
        sanitized_url = InputSanitizer.sanitize_url(url)
        
        # Platform-specific validation
        if platform == "linkedin":
            if not InputValidator.PATTERNS['linkedin_url'].match(sanitized_url):
                raise ValidationError("linkedin_url", 
                    "LinkedIn URL must be in format: https://linkedin.com/in/username")
        
        return sanitized_url
    
    @staticmethod
    def validate_file_upload(filename: str, content_type: str, file_size: int) -> None:
        """Validate file upload parameters"""
        # Validate filename
        if not filename:
            raise ValidationError("filename", "Filename is required")
        
        sanitized_filename = InputSanitizer.sanitize_filename(filename)
        if not sanitized_filename:
            raise ValidationError("filename", "Invalid filename")
        
        # Validate file extension
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_ext not in allowed_extensions:
            raise ValidationError("file_type", 
                f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}")
        
        # Validate content type
        allowed_content_types = {
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain'
        }
        
        if content_type not in allowed_content_types:
            raise ValidationError("content_type", "Invalid file content type")
        
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            raise ValidationError("file_size", f"File size must be less than {max_size // (1024*1024)}MB")
    
    @staticmethod
    def validate_json_structure(data: dict, required_fields: List[str], 
                              optional_fields: List[str] = None) -> dict:
        """Validate JSON structure and required fields"""
        if not isinstance(data, dict):
            raise ValidationError("data", "Data must be a JSON object")
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError("required_fields", 
                f"Missing required fields: {', '.join(missing_fields)}")
        
        # Remove unexpected fields
        allowed_fields = set(required_fields + (optional_fields or []))
        filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        return filtered_data


class SecureInputValidator(BaseModel):
    """Pydantic-based secure input validator"""
    
    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        """Automatically sanitize string inputs"""
        if isinstance(v, str):
            return InputSanitizer.sanitize_text(v)
        return v


# Utility functions
def validate_and_sanitize_input(data: dict, validation_rules: dict) -> dict:
    """Validate and sanitize input data based on rules"""
    validated_data = {}
    
    for field, value in data.items():
        if field in validation_rules:
            rule = validation_rules[field]
            
            try:
                if rule['type'] == 'email':
                    validated_data[field] = InputValidator.validate_email(value)
                elif rule['type'] == 'password':
                    validated_data[field] = InputValidator.validate_password(value)
                elif rule['type'] == 'username':
                    platform = rule.get('platform', 'general')
                    validated_data[field] = InputValidator.validate_username(value, platform)
                elif rule['type'] == 'url':
                    platform = rule.get('platform')
                    validated_data[field] = InputValidator.validate_url(value, platform)
                elif rule['type'] == 'text':
                    validated_data[field] = InputSanitizer.sanitize_text(value)
                elif rule['type'] == 'html':
                    validated_data[field] = InputSanitizer.sanitize_html(value)
                else:
                    validated_data[field] = value
            
            except ValidationError as e:
                logger.warning("Input validation failed", field=field, error=str(e))
                raise e
        else:
            # Default sanitization for unknown fields
            if isinstance(value, str):
                validated_data[field] = InputSanitizer.sanitize_text(value)
            else:
                validated_data[field] = value
    
    return validated_data