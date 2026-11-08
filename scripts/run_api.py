"""Run the FastAPI application with Uvicorn."""

import logging

import uvicorn

from vision_pipeline.api.settings import APISettings


def main() -> None:
    """Start the API server."""
    settings = APISettings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "vision_pipeline.api.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
