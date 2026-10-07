import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from PIL import Image
import json
import xml.etree.ElementTree as ET

from src.utils.logging import setup_logger
from src.data.discovery import DatasetDiscovery, DatasetLocation

logger = setup_logger("visualseek.data.loader")

@dataclass
class ImageRecord:
    image_id: str
    breed: str
    breed_folder: str
    file_path: str
    annotation_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None

class StanfordDogsDataset:
    """Manages loading, validating, and sampling images from the Stanford Dogs Dataset."""

    def __init__(
        self,
        images_dir: Optional[str] = None,
        annotations_dir: Optional[str] = None,
        limit: Optional[int] = None,
        seed: int = 42,
        validate_on_load: bool = False
    ):
        self.limit = limit
        self.seed = seed
        self.validate_on_load = validate_on_load

        # If paths are not provided, run discovery
        if not images_dir:
            discovery = DatasetDiscovery()
            location = discovery.discover()
            self.images_dir = Path(location.images_dir)
            self.annotations_dir = Path(location.annotations_dir) if location.annotations_dir else None
        else:
            self.images_dir = Path(images_dir)
            self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        self.records: List[ImageRecord] = []
        self.corrupt_files: List[str] = []
        self._load_dataset()

    @staticmethod
    def clean_breed_name(breed_folder: str) -> str:
        """Converts breed folder like 'n02085620-Chihuahua' or 'n02099601-golden_retriever' to 'Chihuahua' or 'Golden Retriever'."""
        if "-" in breed_folder:
            name_part = breed_folder.split("-", 1)[1]
        else:
            name_part = breed_folder
        # Replace underscores with spaces and title case
        return name_part.replace("_", " ").title()

    def _load_dataset(self) -> None:
        """Discovers all images and applies stratified sampling if limit is specified."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        all_records_by_breed: Dict[str, List[ImageRecord]] = {}
        valid_extensions = {".jpg", ".jpeg", ".png"}

        breed_dirs = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        total_found = 0

        for b_dir in breed_dirs:
            b_name = self.clean_breed_name(b_dir.name)
            all_records_by_breed[b_name] = []

            annot_b_dir = (self.annotations_dir / b_dir.name) if (self.annotations_dir and (self.annotations_dir / b_dir.name).exists()) else None

            for img_file in sorted(b_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                    img_id = img_file.stem
                    annot_file = (annot_b_dir / img_id) if (annot_b_dir and (annot_b_dir / img_id).exists()) else None

                    rec = ImageRecord(
                        image_id=img_id,
                        breed=b_name,
                        breed_folder=b_dir.name,
                        file_path=str(img_file.resolve()),
                        annotation_path=str(annot_file.resolve()) if annot_file else None
                    )
                    all_records_by_breed[b_name].append(rec)
                    total_found += 1

        logger.info(f"Loaded {total_found} image records across {len(all_records_by_breed)} breeds from {self.images_dir}")

        # Apply sampling if limit is specified
        if self.limit and self.limit < total_found:
            self.records = self._stratified_sample(all_records_by_breed, self.limit)
            logger.info(f"Applied stratified sampling: {len(self.records)} images selected (limit={self.limit})")
        else:
            # Flatten all
            flat = []
            for b_recs in all_records_by_breed.values():
                flat.extend(b_recs)
            self.records = flat

        if self.validate_on_load:
            self.validate_all()

    def _stratified_sample(self, records_by_breed: Dict[str, List[ImageRecord]], target_count: int) -> List[ImageRecord]:
        """Evenly samples records across all breeds using a deterministic seed."""
        rng = random.Random(self.seed)
        num_classes = len(records_by_breed)
        if num_classes == 0:
            return []

        base_per_class = target_count // num_classes
        remainder = target_count % num_classes

        sampled: List[ImageRecord] = []
        for i, (breed, recs) in enumerate(sorted(records_by_breed.items())):
            k = base_per_class + (1 if i < remainder else 0)
            k = min(k, len(recs))
            sampled_recs = rng.sample(recs, k) if len(recs) >= k else recs[:]
            sampled.extend(sampled_recs)

        # Shuffle deterministically
        rng.shuffle(sampled)
        return sampled

    def validate_all(self) -> Tuple[int, int]:
        """Validates all loaded image records to verify PIL readability. Returns (num_valid, num_corrupt)."""
        valid_records = []
        corrupt = []

        for rec in self.records:
            try:
                with Image.open(rec.file_path) as img:
                    img.verify()
                # Re-open to read dimensions since verify() closes image
                with Image.open(rec.file_path) as img:
                    rec.width, rec.height = img.size
                    rec.channels = len(img.getbands())
                valid_records.append(rec)
            except Exception as e:
                logger.warning(f"Corrupted image detected: {rec.file_path} - {e}")
                corrupt.append(rec.file_path)

        self.records = valid_records
        self.corrupt_files = corrupt
        return len(valid_records), len(corrupt)

    def parse_annotation(self, record: ImageRecord) -> Optional[Dict[str, Any]]:
        """Parses Pascal VOC XML annotation file if available."""
        if not record.annotation_path or not Path(record.annotation_path).exists():
            return None
        try:
            tree = ET.parse(record.annotation_path)
            root = tree.getroot()
            boxes = []
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    boxes.append({
                        "xmin": int(bndbox.find("xmin").text),
                        "ymin": int(bndbox.find("ymin").text),
                        "xmax": int(bndbox.find("xmax").text),
                        "ymax": int(bndbox.find("ymax").text)
                    })
            return {"image_id": record.image_id, "breed": record.breed, "bounding_boxes": boxes}
        except Exception as e:
            logger.warning(f"Error parsing annotation {record.annotation_path}: {e}")
            return None

    def get_breed_distribution(self) -> Dict[str, int]:
        """Returns the distribution of images per breed."""
        dist: Dict[str, int] = {}
        for r in self.records:
            dist[r.breed] = dist.get(r.breed, 0) + 1
        return dist

    def to_metadata_list(self) -> List[Dict[str, Any]]:
        """Exports dataset records as a list of dictionaries."""
        return [asdict(r) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ImageRecord:
        return self.records[idx]
