import pytest
from pathlib import Path
from PIL import Image
import tempfile
import os

from src.data.loader import StanfordDogsDataset, ImageRecord
from src.data.discovery import DatasetDiscovery

def test_clean_breed_name():
    assert StanfordDogsDataset.clean_breed_name("n02085620-Chihuahua") == "Chihuahua"
    assert StanfordDogsDataset.clean_breed_name("n02099601-golden_retriever") == "Golden Retriever"
    assert StanfordDogsDataset.clean_breed_name("n02093428-American_Staffordshire_terrier") == "American Staffordshire Terrier"

def test_stratified_sampling():
    # Create fake breed dictionary
    records_by_breed = {
        "BreedA": [ImageRecord(f"A{i}", "BreedA", "n1-BreedA", f"/path/A{i}.jpg") for i in range(20)],
        "BreedB": [ImageRecord(f"B{i}", "BreedB", "n2-BreedB", f"/path/B{i}.jpg") for i in range(20)],
    }
    dataset = StanfordDogsDataset.__new__(StanfordDogsDataset)
    dataset.seed = 42
    sampled = dataset._stratified_sample(records_by_breed, target_count=10)
    
    assert len(sampled) == 10
    breeds = [r.breed for r in sampled]
    assert breeds.count("BreedA") == 5
    assert breeds.count("BreedB") == 5

def test_dataset_discovery_actual():
    discovery = DatasetDiscovery()
    location = discovery.discover()
    assert location.name == "Stanford Dogs"
    assert location.num_breeds == 120
    assert location.num_images == 20580
    assert Path(location.images_dir).exists()

def test_corrupt_image_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        breed_dir = tmp_path / "n02085620-Chihuahua"
        breed_dir.mkdir()
        
        # Write valid image
        valid_img = Image.new("RGB", (50, 50), color="red")
        valid_path = breed_dir / "valid.jpg"
        valid_img.save(valid_path)
        
        # Write corrupt file
        corrupt_path = breed_dir / "corrupt.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_AN_IMAGE_CONTENT")

        ds = StanfordDogsDataset(
            images_dir=str(tmp_path),
            validate_on_load=True
        )
        assert len(ds.records) == 1
        assert len(ds.corrupt_files) == 1
        assert ds.records[0].image_id == "valid"
