import functools
from fastapi import HTTPException
from utils.logger import logger

def handle_exception(func):
    """
    A decorator to catch exceptions in API endpoints, log them,
    and return a proper HTTP 500 error.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # The wrapper must be async to work with FastAPI endpoints
        try:
            # Await the actual async function
            return await func(*args, **kwargs)
        except HTTPException as he:
            # Re-raise HTTPExceptions directly as they are intended for the client
            raise he
        except Exception as e:
            # Log the unexpected error
            logger.error(f"An unexpected error occurred in function '{func.__name__}': {e}", exc_info=True)
            # Raise a standard 500 internal server error
            raise HTTPException(status_code=500, detail="An internal server error occurred.")
    return wrapper

def handle_exception_sync(func):
    """
    A decorator that wraps a SYNCHRONOUS function, logs exceptions,
    and prevents the application from crashing.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Simply call the original synchronous function
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception with a full traceback
            logger.error(f"An uncaught exception occurred in {func.__name__}: {e}", exc_info=True)
            # You might return a default value or re-raise depending on desired behavior
            # For this pipeline, we'll log and stop that specific path.
    return wrapper