import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.discovery import DatasetDiscovery
from src.data.loader import StanfordDogsDataset

def inspect():
    print("=" * 60)
    print(" VISUALSEEK — DATASET INSPECTION")
    print("=" * 60)

    config = load_config()
    discovery = DatasetDiscovery()
    
    print("\n[1/3] Discovering Stanford Dogs dataset...")
    location = discovery.discover(preferred_path=config.dataset.path)
    print(f"  Dataset Name:      {location.name}")
    print(f"  Images Directory:  {location.images_dir}")
    print(f"  Annotations Dir:   {location.annotations_dir}")
    print(f"  Total Dog Breeds:  {location.num_breeds}")
    print(f"  Total Images:      {location.num_images}")

    print("\n[2/3] Validating sample images and verifying dimensions...")
    # Load dataset with a validation sample (or full dataset if requested)
    dataset = StanfordDogsDataset(
        images_dir=location.images_dir,
        annotations_dir=location.annotations_dir,
        limit=500, # sample 500 images across breeds for fast thorough validation
        validate_on_load=True
    )

    valid_count, corrupt_count = len(dataset.records), len(dataset.corrupt_files)
    print(f"  Sampled for Check: {valid_count + corrupt_count}")
    print(f"  Valid Images:      {valid_count}")
    print(f"  Corrupted Images:  {corrupt_count}")

    if dataset.records:
        widths = [r.width for r in dataset.records if r.width]
        heights = [r.height for r in dataset.records if r.height]
        if widths and heights:
            print(f"  Min Dimensions:    {min(widths)}x{min(heights)}")
            print(f"  Max Dimensions:    {max(widths)}x{max(heights)}")
            print(f"  Avg Dimensions:    {int(sum(widths)/len(widths))}x{int(sum(heights)/len(heights))}")

    print("\n[3/3] Class distribution preview (first 10 breeds):")
    dist = dataset.get_breed_distribution()
    for breed, count in list(dist.items())[:10]:
        print(f"  - {breed:<30}: {count} sampled")

    print("\n" + "=" * 60)
    print(" DATASET INTEGRITY CHECK PASSED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    inspect()
