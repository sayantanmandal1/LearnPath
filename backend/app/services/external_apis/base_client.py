"""Base client for external API integrations with retry mechanisms and error handling."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import httpx
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(message)


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(message)


class RetryConfig(BaseModel):
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class BaseAPIClient(ABC):
    """Base class for external API clients with intelligent retry and error handling."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limit_reset: Optional[datetime] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if not self._client:
            headers = self._get_default_headers()
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True
            )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "User-Agent": "AI-Career-Recommender/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers.update(self._get_auth_headers())
            
        return headers
    
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers. Must be implemented by subclasses."""
        pass
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and error handling."""
        await self._ensure_client()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = headers or {}
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Check rate limit
                await self._check_rate_limit()
                
                # Make request
                response = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers
                )
                
                # Handle response
                return await self._handle_response(response)
                
            except RateLimitError as e:
                if attempt == self.retry_config.max_retries:
                    raise
                
                delay = e.retry_after or self._calculate_delay(attempt)
                logger.warning(f"Rate limit hit, retrying in {delay}s")
                await asyncio.sleep(delay)
                
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == self.retry_config.max_retries:
                    raise APIError(f"Request failed after {self.retry_config.max_retries} retries: {str(e)}")
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
                
        raise APIError("Max retries exceeded")
    
    async def _check_rate_limit(self):
        """Check if we're currently rate limited."""
        if self._rate_limit_reset and datetime.utcnow() < self._rate_limit_reset:
            remaining = (self._rate_limit_reset - datetime.utcnow()).total_seconds()
            raise RateLimitError(f"Rate limited, reset in {remaining}s", retry_after=int(remaining))
    
    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and extract data."""
        # Update rate limit info
        self._update_rate_limit_info(response)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            self._rate_limit_reset = datetime.utcnow() + timedelta(seconds=retry_after)
            raise RateLimitError("Rate limit exceeded", retry_after=retry_after)
        
        if response.status_code >= 400:
            error_data = None
            try:
                error_data = response.json()
            except:
                pass
            
            raise APIError(
                f"HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
                response_data=error_data
            )
        
        try:
            return response.json()
        except ValueError:
            return {"content": response.text}
    
    def _update_rate_limit_info(self, response: httpx.Response):
        """Update rate limit information from response headers."""
        # This can be overridden by subclasses for API-specific rate limit headers
        pass
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
            
        return delay
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request."""
        return await self._make_request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request."""
        return await self._make_request("POST", endpoint, data=data)
    
    async def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PUT request."""
        return await self._make_request("PUT", endpoint, data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self._make_request("DELETE", endpoint)