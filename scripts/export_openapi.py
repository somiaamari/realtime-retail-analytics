"""Export the API's generated OpenAPI schema to docs/openapi.json."""

import json
from pathlib import Path

from vision_pipeline.api.main import create_app


def main() -> None:
    """Write a stable, sorted OpenAPI document."""
    destination = Path("docs/openapi.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
