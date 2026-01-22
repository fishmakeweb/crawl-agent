"""
Structured Logging for Code Generation Pipeline

Provides JSON-formatted logging for observability and debugging.
Logs include:
- Query classification decisions
- Generated code
- Execution results
- Data profiles
- Error details
- Performance metrics

Usage:
    from logging_config import get_structured_logger
    
    logger = get_structured_logger("code_generator")
    logger.log_generation_attempt(
        attempt_number=1,
        query="tính trung bình giá",
        generated_code=code,
        execution_result=result
    )
"""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredLogger:
    """
    Wrapper around Python logger for structured JSON logging.
    
    Provides domain-specific log methods for code generation pipeline.
    """
    
    def __init__(self, name: str, base_logger: Optional[logging.Logger] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (usually module name)
            base_logger: Underlying Python logger (default: create new)
        """
        self.name = name
        self.logger = base_logger or logging.getLogger(name)
    
    def _log_json(self, level: int, event_type: str, data: Dict[str, Any]):
        """
        Log structured JSON message.
        
        Args:
            level: Logging level (logging.INFO, logging.ERROR, etc.)
            event_type: Event type identifier
            data: Event data
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            "event_type": event_type,
            **data
        }
        
        # Pretty print for readability in dev, compact in production
        json_str = json.dumps(log_entry, ensure_ascii=False, indent=2)
        self.logger.log(level, json_str)
    
    def log_query_classification(
        self,
        query: str,
        classification_result: Any,
        duration_ms: float = 0.0
    ):
        """Log query classification result"""
        self._log_json(logging.INFO, "query_classification", {
            "query": query,
            "classification": classification_result.query_type.value if hasattr(classification_result, 'query_type') else str(classification_result),
            "confidence": classification_result.confidence if hasattr(classification_result, 'confidence') else 0.0,
            "requires_computation": classification_result.requires_computation if hasattr(classification_result, 'requires_computation') else False,
            "reasoning": classification_result.reasoning if hasattr(classification_result, 'reasoning') else "",
            "duration_ms": duration_ms
        })
    
    def log_data_profile(
        self,
        query: str,
        data_profile: Any,
        product_count: int
    ):
        """Log data profiling results"""
        self._log_json(logging.INFO, "data_profile", {
            "query": query,
            "product_count": product_count,
            "consistency_score": data_profile.consistency_score if hasattr(data_profile, 'consistency_score') else 0.0,
            "field_count": len(data_profile.field_schemas) if hasattr(data_profile, 'field_schemas') else 0,
            "has_consistent_prices": data_profile.has_consistent_prices() if hasattr(data_profile, 'has_consistent_prices') else False,
            "recommendations": data_profile.recommendations if hasattr(data_profile, 'recommendations') else []
        })
    
    def log_generation_attempt(
        self,
        attempt_number: int,
        query: str,
        generated_code: str,
        execution_result: Any,
        error_category: Optional[str] = None
    ):
        """Log code generation attempt"""
        self._log_json(logging.INFO, "generation_attempt", {
            "attempt_number": attempt_number,
            "query": query,
            "code_length": len(generated_code),
            "code_preview": generated_code[:200] + "..." if len(generated_code) > 200 else generated_code,
            "execution_status": execution_result.status.value if hasattr(execution_result, 'status') else str(execution_result),
            "execution_time_ms": execution_result.execution_time_ms if hasattr(execution_result, 'execution_time_ms') else 0.0,
            "success": execution_result.is_success() if hasattr(execution_result, 'is_success') else False,
            "error": execution_result.error if hasattr(execution_result, 'error') else None,
            "error_type": execution_result.error_type if hasattr(execution_result, 'error_type') else None,
            "error_category": error_category
        })
    
    def log_code_execution(
        self,
        function_name: str,
        code_length: int,
        execution_result: Any
    ):
        """Log code execution result"""
        self._log_json(
            logging.INFO if execution_result.is_success() else logging.ERROR,
            "code_execution",
            {
                "function_name": function_name,
                "code_length": code_length,
                "status": execution_result.status.value if hasattr(execution_result, 'status') else str(execution_result),
                "execution_time_ms": execution_result.execution_time_ms if hasattr(execution_result, 'execution_time_ms') else 0.0,
                "success": execution_result.is_success() if hasattr(execution_result, 'is_success') else False,
                "output_keys": list(execution_result.output.keys()) if hasattr(execution_result, 'output') and execution_result.output else [],
                "error": execution_result.error if hasattr(execution_result, 'error') else None
            }
        )
    
    def log_fallback_attempt(
        self,
        query: str,
        fallback_result: Any,
        reason: str = "code_generation_failed"
    ):
        """Log fallback calculator attempt"""
        self._log_json(
            logging.INFO if fallback_result.success else logging.WARNING,
            "fallback_attempt",
            {
                "query": query,
                "reason": reason,
                "method": fallback_result.method if hasattr(fallback_result, 'method') else "unknown",
                "success": fallback_result.success if hasattr(fallback_result, 'success') else False,
                "result": fallback_result.result if hasattr(fallback_result, 'result') else None,
                "error": fallback_result.error if hasattr(fallback_result, 'error') else None
            }
        )
    
    def log_final_result(
        self,
        query: str,
        success: bool,
        method_used: str,
        total_attempts: int,
        total_duration_ms: float = 0.0,
        error_message: Optional[str] = None
    ):
        """Log final pipeline result"""
        self._log_json(
            logging.INFO if success else logging.ERROR,
            "final_result",
            {
                "query": query,
                "success": success,
                "method_used": method_used,
                "total_attempts": total_attempts,
                "total_duration_ms": total_duration_ms,
                "error_message": error_message
            }
        )
    
    def log_error_classification(
        self,
        error_type: str,
        error_category: str,
        feedback: str,
        is_retryable: bool
    ):
        """Log error classification decision"""
        self._log_json(logging.WARNING, "error_classification", {
            "error_type": error_type,
            "error_category": error_category,
            "feedback": feedback,
            "is_retryable": is_retryable
        })
    
    # Delegate standard logging methods
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)


# Global registry of structured loggers
_loggers: Dict[str, StructuredLogger] = {}


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger by name.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
):
    """
    Configure global logging settings.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Optional custom format string
    """
    if format_string is None:
        # Default format for structured logging
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Auto-configure on import
configure_logging()
