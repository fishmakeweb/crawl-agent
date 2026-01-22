"""
Safe Code Execution Sandbox using RestrictedPython

Executes LLM-generated Python functions in a restricted environment:
- Blocks dangerous imports (os, sys, subprocess, eval, exec)
- Whitelists safe builtins (len, sum, max, min, sorted, etc.)
- Enforces timeout and memory limits using resource module (Linux)
- Returns structured execution results with errors

Trade-offs:
- RestrictedPython: High security, fast, medium complexity (RECOMMENDED)
- exec() + globals: Medium security, very fast, low complexity
- Docker/gVisor: Very high security, slow, high complexity (only for multi-tenant)
"""
import os
import sys
import logging
import json
import re
import statistics
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import signal
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import RestrictedPython (if not available, fallback to exec)
try:
    from RestrictedPython import compile_restricted, safe_globals, limited_builtins, utility_builtins
    RESTRICTED_PYTHON_AVAILABLE = True
    logger.info("RestrictedPython available - using secure sandbox")
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logger.warning("RestrictedPython not installed - using exec() fallback (less secure)")


class ExecutionStatus(Enum):
    """Execution result status"""
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    INVALID_OUTPUT = "invalid_output"


@dataclass
class ExecutionResult:
    """Result of code execution"""
    status: ExecutionStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS
    
    def get_error_message(self) -> str:
        """Get user-friendly error message"""
        if self.status == ExecutionStatus.SUCCESS:
            return ""
        
        if self.status == ExecutionStatus.TIMEOUT:
            return "Function execution timed out (>5s). Infinite loop or too slow."
        elif self.status == ExecutionStatus.MEMORY_LIMIT:
            return "Function exceeded memory limit."
        elif self.status == ExecutionStatus.SYNTAX_ERROR:
            return f"Syntax error in generated code: {self.error}"
        elif self.status == ExecutionStatus.RUNTIME_ERROR:
            return f"Runtime error: {self.error_type}: {self.error}"
        elif self.status == ExecutionStatus.INVALID_OUTPUT:
            return f"Invalid output format: {self.error}"
        else:
            return f"Unknown error: {self.error}"


class TimeoutException(Exception):
    """Raised when execution times out"""
    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for execution timeout using signal (Linux only)"""
    def timeout_handler(signum, frame):
        raise TimeoutException("Execution timed out")
    
    # Only works on Unix-like systems
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # On Windows, no timeout (or use threading alternative)
        logger.warning("Signal-based timeout not available on this platform")
        yield


class SafeExecutionSandbox:
    """
    Safe sandbox for executing LLM-generated analysis functions.
    
    Features:
    - RestrictedPython compilation (blocks dangerous AST nodes)
    - Whitelisted builtins only
    - Resource limits (timeout, memory)
    - Structured error reporting
    
    Usage:
        sandbox = SafeExecutionSandbox(timeout_seconds=5)
        result = sandbox.execute_function(code_str, "analyze", products_data)
        if result.is_success():
            print(result.output)
        else:
            print(result.get_error_message())
    """
    
    # Whitelisted safe builtins
    SAFE_BUILTINS = {
        # Type constructors
        'dict', 'list', 'tuple', 'set', 'frozenset', 'str', 'int', 'float', 'bool',
        
        # Built-in functions
        'len', 'sum', 'max', 'min', 'abs', 'round', 'sorted', 'reversed',
        'enumerate', 'zip', 'map', 'filter', 'any', 'all',
        'range', 'slice',
        
        # Type checking
        'isinstance', 'type', 'hasattr', 'getattr',
        
        # Utilities
        'print',  # For debugging (output will be captured)
    }
    
    # Whitelisted modules (limited imports)
    SAFE_MODULES = {
        're': re,
        'json': json,
        'statistics': statistics,
    }
    
    def __init__(
        self,
        timeout_seconds: int = 5,
        max_memory_mb: Optional[int] = 256,
        use_restricted_python: bool = True
    ):
        """
        Initialize sandbox.
        
        Args:
            timeout_seconds: Max execution time (default 5s)
            max_memory_mb: Max memory usage in MB (Linux only, None to disable)
            use_restricted_python: Use RestrictedPython if available
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.use_restricted_python = use_restricted_python and RESTRICTED_PYTHON_AVAILABLE
        
        # Set resource limits (Linux only)
        if max_memory_mb and hasattr(os, 'fork'):  # Unix-like system
            try:
                import resource
                # Set memory limit (virtual memory)
                max_memory_bytes = max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
                logger.info(f"Memory limit set to {max_memory_mb}MB")
            except Exception as e:
                logger.warning(f"Could not set memory limit: {e}")
    
    def _build_safe_globals(self) -> Dict[str, Any]:
        """Build safe globals dict with whitelisted builtins and modules"""
        if self.use_restricted_python:
            # Use RestrictedPython's safe_globals as base
            safe_dict = safe_globals.copy()
            
            # Add limited builtins
            safe_dict['__builtins__'] = {
                name: __builtins__[name] 
                for name in self.SAFE_BUILTINS 
                if name in __builtins__
            }
        else:
            # Manual whitelist
            safe_dict = {
                '__builtins__': {
                    name: getattr(__builtins__, name) 
                    for name in self.SAFE_BUILTINS 
                    if hasattr(__builtins__, name)
                }
            }
        
        # Add safe modules
        safe_dict.update(self.SAFE_MODULES)
        
        return safe_dict
    
    def _compile_code(self, code: str) -> tuple[Any, Optional[str]]:
        """
        Compile code using RestrictedPython or standard compile.
        
        Returns:
            (compiled_code, error_message)
        """
        try:
            if self.use_restricted_python:
                # RestrictedPython compilation (AST-based security)
                result = compile_restricted(code, '<string>', 'exec')
                
                if result.errors:
                    error_msg = "; ".join(result.errors)
                    return None, f"RestrictedPython errors: {error_msg}"
                
                return result.code, None
            else:
                # Standard Python compilation
                compiled = compile(code, '<string>', 'exec')
                return compiled, None
                
        except SyntaxError as e:
            return None, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return None, str(e)
    
    def execute_function(
        self,
        code: str,
        function_name: str,
        *args,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a generated function with arguments.
        
        Args:
            code: Python code containing the function definition
            function_name: Name of function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            ExecutionResult with status and output/error
        """
        import time
        start_time = time.time()
        
        # Step 1: Compile code
        compiled_code, compile_error = self._compile_code(code)
        if compile_error:
            logger.error(f"Compilation failed: {compile_error}")
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error=compile_error,
                error_type="SyntaxError"
            )
        
        # Step 2: Prepare safe execution environment
        safe_globals = self._build_safe_globals()
        safe_locals = {}
        
        # Step 3: Execute code to define function
        try:
            with timeout_context(self.timeout_seconds):
                exec(compiled_code, safe_globals, safe_locals)
        except TimeoutException:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Function definition timed out after {self.timeout_seconds}s",
                execution_time_ms=execution_time
            )
        except MemoryError:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.MEMORY_LIMIT,
                error=f"Exceeded memory limit of {self.max_memory_mb}MB",
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error executing code definition: {type(e).__name__}: {e}")
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time
            )
        
        # Step 4: Check if function exists
        if function_name not in safe_locals:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error=f"Function '{function_name}' not found in code",
                error_type="NameError",
                execution_time_ms=execution_time
            )
        
        func = safe_locals[function_name]
        
        # Step 5: Call function with arguments
        try:
            with timeout_context(self.timeout_seconds):
                result = func(*args, **kwargs)
        except TimeoutException:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Function execution timed out after {self.timeout_seconds}s",
                execution_time_ms=execution_time
            )
        except MemoryError:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.MEMORY_LIMIT,
                error=f"Exceeded memory limit of {self.max_memory_mb}MB",
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Runtime error in function: {type(e).__name__}: {e}")
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time
            )
        
        # Step 6: Validate output is JSON-serializable dict
        if not isinstance(result, dict):
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.INVALID_OUTPUT,
                error=f"Function must return dict, got {type(result).__name__}",
                execution_time_ms=execution_time
            )
        
        # Try to serialize to ensure JSON-compatible
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                status=ExecutionStatus.INVALID_OUTPUT,
                error=f"Output not JSON-serializable: {str(e)}",
                execution_time_ms=execution_time
            )
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Function executed successfully in {execution_time:.2f}ms")
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=result,
            execution_time_ms=execution_time
        )


# Singleton instance
_sandbox: Optional[SafeExecutionSandbox] = None

def get_sandbox(
    timeout_seconds: int = 5,
    max_memory_mb: Optional[int] = 256
) -> SafeExecutionSandbox:
    """Get or create singleton sandbox instance"""
    global _sandbox
    if _sandbox is None:
        _sandbox = SafeExecutionSandbox(
            timeout_seconds=timeout_seconds,
            max_memory_mb=max_memory_mb
        )
    return _sandbox
