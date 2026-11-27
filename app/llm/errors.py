"""Errors of the LLM client. Providers translate their SDK errors into these."""


class LLMError(Exception):
    """Base class for every LLM client error."""


class LLMTransientError(LLMError):
    """A temporary failure (timeout, rate limit, server error); the client retries it."""


class LLMTimeoutError(LLMTransientError):
    """No response within the configured timeout."""


class LLMPermanentError(LLMError):
    """A failure that retrying cannot fix (bad request, auth, invalid schema, empty answer)."""


class LLMRetriesExhaustedError(LLMError):
    """Every attempt failed with a temporary error."""

    def __init__(self, attempts: int, last_error: LLMError) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM call failed after {attempts} attempts; last error: {type(last_error).__name__}: {last_error}"
        )
