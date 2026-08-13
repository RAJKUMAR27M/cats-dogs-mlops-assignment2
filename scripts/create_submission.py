"""Create a zip archive of the project for submission (≤ 10 MB)."""

import argparse
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Top-level directories to skip (matched only at depth 1)
SKIP_TOP_DIRS = {"data", "artifacts", "logs", "mlruns", "mlartifacts", "dvc_store"}
# sample_data/ is intentionally NOT in SKIP_TOP_DIRS so it gets included

# Directory names to skip at any depth
SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", "env", ".pytest_cache", "dist", "build"}

SKIP_EXTS = {".h5", ".pb", ".pkl", ".pyc", ".pyo", ".zip"}
SKIP_FILES = {"kaggle.json", ".env", ".env.local", "submission.zip"}


def _should_skip(rel: Path) -> bool:
    parts = rel.parts
    # Block top-level dirs (e.g. data/, artifacts/) but allow src/data/
    if len(parts) > 0 and parts[0] in SKIP_TOP_DIRS:
        return True
    for part in parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    if rel.suffix in SKIP_EXTS:
        return True
    if rel.name in SKIP_FILES:
        return True
    return False


def create_zip(project_dir: str = ".", output_file: str = "submission.zip") -> None:
    project_path = Path(project_dir).resolve()
    output_path = Path(output_file).resolve()
    count = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(project_path.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(project_path)
            if _should_skip(rel):
                continue
            zf.write(f, str(rel))
            count += 1

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Created %s: %d files, %.2f MB", output_file, count, size_mb)
    if size_mb > 10:
        logger.warning("ZIP exceeds 10 MB limit (%.2f MB) – review excluded paths", size_mb)


def main():
    parser = argparse.ArgumentParser(description="Create submission zip")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--output", default="submission.zip")
    args = parser.parse_args()
    create_zip(args.project_dir, args.output)


if __name__ == "__main__":
    main()
