from pathlib import Path
import os
from typing import Optional

def get_project_root() -> Path:
    """Returns the root directory of the VisualSeek project."""
    # This file is located at <project_root>/src/utils/paths.py
    return Path(__file__).resolve().parent.parent.parent

def get_artifacts_dir(custom_path: Optional[str] = None) -> Path:
    """Returns the artifacts directory, checking environment variable or custom config."""
    env_dir = os.environ.get("VISUALSEEK_ARTIFACTS_DIR")
    if env_dir:
        path = Path(env_dir)
    elif custom_path:
        path = Path(custom_path) if Path(custom_path).is_absolute() else get_project_root() / custom_path
    else:
        path = get_project_root() / "artifacts"
    
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dir(path: Path) -> Path:
    """Ensures a directory exists, creating parents if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path
