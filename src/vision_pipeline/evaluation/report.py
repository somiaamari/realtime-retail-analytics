"""Environment capture and combined Markdown/JSON benchmark reports."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _package_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_git_hash(root: Path) -> str | None:
    """Read the current commit from .git metadata without invoking Git."""
    git_dir = root / ".git"
    if git_dir.is_file():
        try:
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                git_dir = (root / pointer.split(":", 1)[1].strip()).resolve()
        except OSError:
            return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref:
                        return commit
    except (OSError, ValueError):
        return None
    return os.environ.get("GITHUB_SHA")


def collect_environment(
    *,
    repository_root: str | Path,
    config_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Capture reproducibility metadata and hashes for the current run."""
    root = Path(repository_root)
    config_hash = hashlib.sha256()
    for path in sorted((Path(item) for item in config_paths or []), key=str):
        resolved_path = path.resolve()
        try:
            display_path = resolved_path.relative_to(root.resolve())
        except ValueError:
            display_path = Path(path.name)
        config_hash.update(str(display_path).encode())
        config_hash.update(resolved_path.read_bytes())
    ram_bytes = None
    gpu_name = None
    try:
        psutil = importlib.import_module("psutil")
        ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_model": platform.processor() or platform.machine(),
        "ram_bytes": ram_bytes,
        "gpu": gpu_name,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "ultralytics": _package_version("ultralytics"),
        "git_commit": _local_git_hash(root),
        "config_sha256": config_hash.hexdigest(),
    }


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and available PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_hardware_result(
    results: dict[str, Any],
    directory: str | Path,
    environment: dict[str, Any],
) -> Path:
    """Save a compact result JSON with a dated hardware-specific filename."""
    cpu = str(environment.get("cpu_model") or "unknown")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", cpu).strip("-").lower()[:48] or "unknown"
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    destination = Path(directory) / f"{day}_{slug}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment, "results": results}
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return destination


def update_baseline_markdown(
    path: str | Path,
    *,
    model: str,
    device: str,
    imgsz: int,
    result: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    """Append a measured before-optimization row to the Phase 7 baseline table."""
    baseline = Path(path)
    baseline.parent.mkdir(parents=True, exist_ok=True)
    if not baseline.exists():
        baseline.write_text(
            "# Phase 6 system baseline\n\n"
            "Measured, unoptimized results. Phase 7 should add optimized results "
            "beside these rows and preserve the original measurements.\n\n"
            "| Model | Device | imgsz | FPS | Mean / p95 end-to-end (ms) | "
            "Date | Hardware | Optimized FPS | Optimized p95 (ms) |\n"
            "|---|---|---:|---:|---:|---|---|---:|---:|\n",
            encoding="utf-8",
        )
    end_to_end = result["end_to_end"]
    date = str(environment.get("timestamp_utc", ""))[:10]
    hardware = str(environment.get("cpu_model") or "unknown").replace("|", "/")
    baseline.write_text(
        baseline.read_text(encoding="utf-8")
        + (
            f"| {model} | {device} | {imgsz} | "
            f"{_format_value(end_to_end.get('fps'))} | "
            f"{_format_value(end_to_end.get('mean_ms'))} / "
            f"{_format_value(end_to_end.get('p95_ms'))} | "
            f"{date} | {hardware} | N/A | N/A |\n"
        ),
        encoding="utf-8",
    )


def _format_value(value: Any) -> str:
    """Format numbers consistently for human-readable result tables."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def _target_value(results: dict[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted target path in nested benchmark results."""
    value: Any = results
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _render_targets(
    results: dict[str, Any],
    targets: dict[str, Any],
) -> list[str]:
    """Render pass/fail statuses from target entries with metric/op/value."""
    lines = [
        "## Target summary",
        "",
        "| Metric | Target | Result | Status |",
        "|---|---:|---:|---|",
    ]
    for name, specification in targets.items():
        if not isinstance(specification, dict):
            continue
        metric = str(specification.get("metric", name))
        operator = str(specification.get("operator", ">="))
        target = specification.get("value")
        result = _target_value(results, metric)
        passed = False
        if result is not None and isinstance(target, (int, float)):
            if operator == ">=":
                passed = result >= target
            elif operator == "<=":
                passed = result <= target
            elif operator == ">":
                passed = result > target
            elif operator == "<":
                passed = result < target
        status = "PENDING" if result is None else "PASS" if passed else "FAIL"
        lines.append(
            f"| {metric} | {operator} {_format_value(target)} | "
            f"{_format_value(result)} | {status} |"
        )
    return lines + [""]


def write_phase6_report(
    output_directory: str | Path,
    *,
    environment: dict[str, Any],
    results: dict[str, Any],
    targets: dict[str, Any] | None = None,
) -> Path:
    """Write combined ``REPORT.md`` and ``results.json`` files."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(
        json.dumps(
            {"environment": environment, "results": results}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    for section, payload in results.items():
        (destination / f"{section}.json").write_text(
            json.dumps(
                {"environment": environment, "results": payload},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    lines = [
        "# Phase 6: Evaluation and benchmark report",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in environment.items():
        lines.append(f"| {key.replace('_', ' ').title()} | {_format_value(value)} |")
    for section in (
        "detection",
        "tracking",
        "analytics",
        "performance",
        "sweep",
        "error_analysis",
    ):
        payload = results.get(section)
        lines.extend(("", f"## {section.replace('_', ' ').title()}", ""))
        if not isinstance(payload, dict) or payload.get("status") == "pending":
            reason = (
                payload.get("reason", "No benchmark data supplied.")
                if isinstance(payload, dict)
                else "No benchmark data supplied."
            )
            lines.extend((f"**Pending:** {reason}", ""))
            continue
        metrics = payload.get("metrics", payload)
        if isinstance(metrics, dict):
            lines.extend(("| Metric | Value |", "|---|---:|"))
            for name, value in metrics.items():
                if isinstance(value, (dict, list)):
                    continue
                lines.append(f"| {name.replace('_', ' ')} | {_format_value(value)} |")
            lines.append("")
        chart = payload.get("chart")
        if chart is None and isinstance(payload.get("artifacts"), dict):
            chart = payload["artifacts"].get("chart")
        chart_paths = payload.get("charts", [])
        if chart:
            chart_paths = [chart, *chart_paths]
        for chart_path in chart_paths:
            if chart_path:
                lines.extend((f"![{section} chart]({chart_path})", ""))
        if section == "error_analysis":
            lines.extend(("### Frequent failure frames", ""))
            lines.extend(("| Sequence | Frame | Failure count |", "|---|---:|---:|"))
            for sequence in payload.get("sequences", []):
                for failure in sequence.get("top_failure_frames", []):
                    lines.append(
                        f"| {sequence.get('sequence', 'sequence')} | "
                        f"{failure['frame']} | {failure['failure_count']} |"
                    )
            lines.append("")
        if section == "error_analysis":
            weaknesses = (
                payload.get("known_weaknesses") or payload.get("limitations") or []
            )
            lines.extend(("### Known weaknesses", ""))
            lines.extend(f"- {item}" for item in weaknesses)
            lines.append("")
    if targets is not None:
        lines.extend(_render_targets(results, targets))
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return destination / "REPORT.md"


def load_targets(path: str | Path) -> dict[str, Any]:
    """Load and validate the evaluation target mapping."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation targets must be a mapping: {path}")
    return payload
