"""PCG render exceptions for better error handling and recovery."""

import logging

logger = logging.getLogger(__name__)


class GradioRenderFailure(Exception):
    """
    Raised when Gradio PCG render service call fails.
    
    This exception encapsulates different failure modes:
    - Timeout: Service response takes >150 seconds
    - HTTP errors: 403, 500, 502, 503, etc.
    - Network errors: Connection refused, socket timeout, DNS failure
    - Format errors: Malformed response, invalid image paths
    - Invalid input: Bad actor data, out-of-range parameters
    
    Some failure modes are retryable (transient), others are permanent.
    """
    
    def __init__(self, reason, details="", actor_count=0, attempt=1):
        """
        Initialize GradioRenderFailure exception.
        
        Args:
            reason (str): Error category - one of:
                - "timeout": Request exceeded max_timeout_seconds
                - "http_error": HTTP status error (403, 500, 502, 503)
                - "connect_error": Network connection failure
                - "format_error": Malformed response from service
                - "json_error": JSON parsing failed
                - "malformed_response": Response structure invalid
                - "invalid_image_path": Image path doesn't exist
                - "invalid_actors": Invalid actor data
                - "unknown": Unexpected error
            
            details (str): Error message details from the underlying exception
            actor_count (int): Number of actors in the scene (for context)
            attempt (int): Which retry attempt this was (1-3)
        """
        self.reason = reason
        self.details = details
        self.actor_count = actor_count
        self.attempt = attempt
        
        # Build descriptive error message
        msg = f"[PCG Render Failure] {reason}"
        
        if attempt > 1:
            msg += f" (attempt {attempt}/3)"
        
        if actor_count > 0:
            msg += f" (actors={actor_count})"
        
        if details:
            msg += f": {details}"
        
        super().__init__(msg)
    
    def is_retryable(self):
        """
        Determine if this error is transient and might succeed on retry.
        
        Transient failures (retryable):
        - timeout: Service was slow, might recover
        - connect_error: Network blip, might recover
        - http_error: Service overload (502, 503), might recover
        
        Permanent failures (non-retryable):
        - format_error: Bad service implementation
        - json_error: Bad service implementation
        - malformed_response: Bad service implementation
        - invalid_actors: Bad input from us
        - invalid_image_path: Service configuration issue
        
        Returns:
            bool: True if retry is likely to help, False if error is structural
        """
        retryable_reasons = {
            "timeout",          # Service slow or overloaded
            "connect_error",    # Transient network failure
            "http_error",       # Server error, might recover
        }
        
        return self.reason in retryable_reasons
    
    def __repr__(self):
        """Return detailed representation for debugging."""
        return (
            f"GradioRenderFailure("
            f"reason={self.reason!r}, "
            f"attempt={self.attempt}, "
            f"actors={self.actor_count}, "
            f"details={self.details!r})"
        )


class PCGServiceUnavailable(GradioRenderFailure):
    """PCG service is unreachable or crashed."""
    
    def __init__(self, details="", attempt=1):
        super().__init__(
            reason="service_unavailable",
            details=details,
            attempt=attempt
        )
    
    def is_retryable(self):
        """Service unavailable is transient in nature."""
        return True


class PCGTimeout(GradioRenderFailure):
    """PCG render request timed out."""
    
    def __init__(self, timeout_seconds, details="", actor_count=0, attempt=1):
        self.timeout_seconds = timeout_seconds
        details = f"Timeout after {timeout_seconds}s: {details}" if details else f"Timeout after {timeout_seconds}s"
        super().__init__(
            reason="timeout",
            details=details,
            actor_count=actor_count,
            attempt=attempt
        )
    
    def is_retryable(self):
        """Timeouts are transient."""
        return True


class PCGHTTPError(GradioRenderFailure):
    """PCG service returned HTTP error."""
    
    def __init__(self, status_code, details="", attempt=1):
        self.status_code = status_code
        super().__init__(
            reason="http_error",
            details=f"HTTP {status_code}: {details}" if details else f"HTTP {status_code}",
            attempt=attempt
        )
    
    def is_retryable(self):
        """5xx errors are transient, 4xx are usually permanent."""
        # 502, 503, 504 are retryable (bad gateway, service unavailable, gateway timeout)
        # 500 is retryable (internal server error, might be transient)
        retryable_codes = {500, 502, 503, 504}
        return self.status_code in retryable_codes


class PCGFormatError(GradioRenderFailure):
    """PCG service returned malformed response."""
    
    def __init__(self, details="", attempt=1):
        super().__init__(
            reason="format_error",
            details=f"Malformed response: {details}" if details else "Malformed response",
            attempt=attempt
        )
    
    def is_retryable(self):
        """Format errors are usually permanent (bad service logic)."""
        return False
