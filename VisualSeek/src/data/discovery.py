import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
import json

from src.utils.logging import setup_logger
from src.utils.paths import get_project_root

logger = setup_logger("visualseek.data.discovery")

@dataclass
class DatasetLocation:
    name: str
    images_dir: str
    annotations_dir: Optional[str]
    num_breeds: int
    num_images: int
    breeds: List[str]

class DatasetDiscovery:
    """Discovers and validates the Stanford Dogs dataset dynamically without hardcoding nested paths."""

    STANFORD_BREED_PATTERN = re.compile(r"^n\d{7}-[\w-]+$")

    def __init__(self, search_roots: Optional[List[Path]] = None):
        if search_roots is None:
            # Default search roots: parent directory of VisualSeek, and current project
            proj_root = get_project_root()
            self.search_roots = [
                proj_root.parent,
                proj_root,
                Path("D:/Hack-O-Week-OddSem-(2026-27)"),
                Path.cwd()
            ]
        else:
            self.search_roots = [Path(p) for p in search_roots]

    def discover(self, preferred_path: Optional[str] = None) -> DatasetLocation:
        """Finds the dataset location by examining preferred path or crawling search roots."""
        if preferred_path:
            p = Path(preferred_path)
            if p.exists():
                loc = self._evaluate_candidate_dir(p)
                if loc:
                    return loc

        # Check search roots
        for root in self.search_roots:
            if not root.exists():
                continue
            
            logger.info(f"Scanning candidate search directory: {root}")
            # First check direct directory or immediate subdirectories like 'archive (1)', 'StanfordDogs', 'images', etc.
            found = self._find_dataset_in_tree(root, max_depth=3)
            if found:
                logger.info(f"Discovered Stanford Dogs dataset at: {found.images_dir} ({found.num_images} images across {found.num_breeds} breeds)")
                return found

        raise FileNotFoundError(
            "Could not locate Stanford Dogs dataset in candidate directories. "
            "Please ensure the dataset is extracted or set VISUALSEEK_DATASET_DIR environment variable."
        )

    def _find_dataset_in_tree(self, root: Path, max_depth: int = 3) -> Optional[DatasetLocation]:
        """Walks directories up to max_depth looking for Stanford Dogs image directories."""
        # Check root itself
        loc = self._evaluate_candidate_dir(root)
        if loc:
            return loc

        # Breadth-first / shallow search
        try:
            for current_root, dirs, _ in os.walk(root):
                depth = len(Path(current_root).relative_to(root).parts)
                if depth > max_depth:
                    dirs.clear()
                    continue

                for d in list(dirs):
                    cand = Path(current_root) / d
                    loc = self._evaluate_candidate_dir(cand)
                    if loc:
                        return loc
        except Exception as e:
            logger.warning(f"Error while scanning {root}: {e}")

        return None

    def _evaluate_candidate_dir(self, directory: Path) -> Optional[DatasetLocation]:
        """Determines if a directory contains Stanford Dogs breed subfolders."""
        if not directory.is_dir():
            return None

        # Check if this directory directly has subfolders matching Stanford Dogs breed patterns
        try:
            subdirs = [p for p in directory.iterdir() if p.is_dir()]
        except Exception:
            return None

        matched_breeds = [
            d.name for d in subdirs
            if self.STANFORD_BREED_PATTERN.match(d.name) or ("-" in d.name and d.name.startswith("n02"))
        ]

        # Stanford dogs has 120 breeds. If we have at least 10 matching breed folders, this is the breed root.
        if len(matched_breeds) >= 10:
            num_images = 0
            for b in subdirs:
                if b.name in matched_breeds:
                    num_images += sum(1 for f in b.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"])

            # Look for annotations directory nearby
            annot_dir = self._find_annotation_dir_nearby(directory)

            return DatasetLocation(
                name="Stanford Dogs",
                images_dir=str(directory.resolve()),
                annotations_dir=str(annot_dir.resolve()) if annot_dir else None,
                num_breeds=len(matched_breeds),
                num_images=num_images,
                breeds=sorted(matched_breeds)
            )

        # Check if there is an 'Images' or 'images' subfolder
        for sub in subdirs:
            if sub.name.lower() == "images":
                nested_loc = self._evaluate_candidate_dir(sub)
                if nested_loc:
                    return nested_loc

        return None

    def _find_annotation_dir_nearby(self, images_dir: Path) -> Optional[Path]:
        """Tries to find matching Annotations/Annotation folder adjacent to or in parents of images_dir."""
        # Check sibling directories of images_dir or parent's siblings
        candidates = [
            images_dir.parent / "Annotation",
            images_dir.parent / "annotation",
            images_dir.parent / "annotations" / "Annotation",
            images_dir.parent.parent / "annotations" / "Annotation",
            images_dir.parent.parent / "Annotation"
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c
        return None

    def save_discovery_cache(self, location: DatasetLocation, cache_path: Path) -> None:
        """Saves discovery metadata to JSON cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(asdict(location), f, indent=2)

    def load_discovery_cache(self, cache_path: Path) -> Optional[DatasetLocation]:
        """Loads discovery metadata from JSON cache if valid."""
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loc = DatasetLocation(**data)
                if Path(loc.images_dir).exists():
                    return loc
            except Exception as e:
                logger.warning(f"Failed to read discovery cache from {cache_path}: {e}")
        return None
