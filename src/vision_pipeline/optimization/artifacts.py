"""SHA256 manifest registration for release-published model artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest for a model file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_artifact(
    weights_path: str | Path,
    artifact_path: str | Path,
    precision: str,
    *,
    registry_path: str | Path = "models/weights.json",
) -> None:
    """Update the local release manifest without copying model binaries."""
    weights = Path(weights_path)
    artifact = Path(artifact_path)
    registry = Path(registry_path)
    document: dict[str, Any] = (
        json.loads(registry.read_text(encoding="utf-8"))
        if registry.is_file()
        else {
            "schema_version": 1,
            "repository": "somiaamari/realtime-retail-analytics",
            "models": {},
        }
    )
    model_key = weights.stem
    source_record: dict[str, str] = {"filename": weights.name}
    if weights.is_file():
        source_record["sha256"] = sha256_file(weights)
    model_entry = document.setdefault("models", {}).setdefault(
        model_key,
        {
            "source_weights": source_record,
            "artifacts": {},
        },
    )
    filename = artifact.name
    model_entry.setdefault("artifacts", {})[precision] = {
        "filename": filename,
        "sha256": sha256_file(artifact),
        "release_url": (
            "https://github.com/somiaamari/realtime-retail-analytics/releases/"
            f"latest/download/{filename}"
        ),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
