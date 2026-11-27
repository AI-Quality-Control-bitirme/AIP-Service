import pytest


@pytest.fixture
def anyio_backend():
    # Async tests use the anyio pytest plugin (@pytest.mark.anyio), on asyncio only.
    return "asyncio"
