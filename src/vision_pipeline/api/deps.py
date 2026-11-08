"""FastAPI dependencies for authentication and database sessions."""

import secrets
from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session


def get_session(request: Request) -> Iterator[Session]:
    """Provide one request-scoped SQLAlchemy session."""
    factory = request.app.state.session_factory
    with factory() as session:
        yield session


def require_api_key(request: Request) -> None:
    """Validate X-API-Key when API-key authentication is enabled."""
    configured = request.app.state.settings.api_key
    if configured is None:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(supplied, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unauthorized", "message": "A valid API key is required."},
        )


Authenticated = Depends(require_api_key)
