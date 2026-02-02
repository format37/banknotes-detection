#!/usr/bin/env python3
"""
Convert PASCAL VOC XML annotations to Hugging Face ImageFolder format.

This script converts the banknotes dataset from PASCAL VOC XML format
to Hugging Face's ImageFolder format with metadata.jsonl files.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Class mapping (0-indexed)
CLASS_NAMES = [
    "check",
    "napkins",
    "glue",
    "banknotes",
    "banknotes_batch",
    "calculator",
    "bcard",
    "partpack",
    "photo_paper",
    "iceberg_card",
    "document",
    "package",
    "smartphone",
    "svetocopy",
    "banknotes10",
    "banknotes50",
    "banknotes100",
    "banknotes200",
    "banknotes500",
    "banknotes1000",
    "banknotes2000",
    "banknotes5000",
    "mouse",
    "other",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse PASCAL VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Get filename
    filename = root.find("filename").text

    # Parse objects
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text

        # Skip unknown classes
        if name not in CLASS_TO_ID:
            print(f"Warning: Unknown class '{name}' in {xml_path}")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Convert from PASCAL VOC [xmin, ymin, xmax, ymax] to COCO [x, y, width, height]
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        objects.append({
            "bbox": [xmin, ymin, bbox_width, bbox_height],
            "category": CLASS_TO_ID[name],
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def convert_split(source_dir: Path, output_dir: Path, split_name: str):
    """Convert a single split (train or test) to ImageFolder format."""
    source_split = source_dir / split_name
    output_split = output_dir / split_name

    # Create output directory
    output_split.mkdir(parents=True, exist_ok=True)

    metadata = []
    image_count = 0

    # Process all XML files
    xml_files = sorted(source_split.glob("*.xml"))

    for xml_path in xml_files:
        annotation = parse_voc_xml(xml_path)

        # Find corresponding image
        image_name = annotation["filename"]
        image_path = source_split / image_name

        if not image_path.exists():
            # Try common extensions
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                alt_path = source_split / (xml_path.stem + ext)
                if alt_path.exists():
                    image_path = alt_path
                    image_name = alt_path.name
                    break

        if not image_path.exists():
            print(f"Warning: Image not found for {xml_path}")
            continue

        # Copy image to output directory
        output_image_path = output_split / image_name
        shutil.copy2(image_path, output_image_path)

        # Create metadata entry
        metadata_entry = {
            "file_name": image_name,
            "width": annotation["width"],
            "height": annotation["height"],
            "objects": annotation["objects"],
        }
        metadata.append(metadata_entry)
        image_count += 1

    # Write metadata.jsonl
    metadata_path = output_split / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Converted {image_count} images for {split_name} split")
    return image_count


def create_dataset_readme(output_dir: Path, train_count: int, test_count: int):
    """Create the dataset README.md (dataset card)."""
    readme_content = '''---
license: mit
task_categories:
  - object-detection
tags:
  - object-detection
  - banknotes
  - currency
  - russian-rubles
  - pascal-voc
  - coco-format
size_categories:
  - 1K<n<10K
---

# Russian Rubles Banknotes Object Detection Dataset

A dataset for detecting Russian ruble banknotes and common office items in images.

## Dataset Summary

- **Total images**: {total_count} ({train_count} train / {test_count} test)
- **Format**: ImageFolder with COCO-style bounding box annotations
- **Image format**: JPEG
- **Annotation format**: `[x, y, width, height]` (COCO format)
- **Classes**: 24 object categories

## Class Labels

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | check | Check/receipt |
| 1 | napkins | Napkins |
| 2 | glue | Glue |
| 3 | banknotes | Generic banknotes |
| 4 | banknotes_batch | Batch of banknotes |
| 5 | calculator | Calculator |
| 6 | bcard | Business card |
| 7 | partpack | Part pack |
| 8 | photo_paper | Photo paper |
| 9 | iceberg_card | Iceberg card |
| 10 | document | Document |
| 11 | package | Package |
| 12 | smartphone | Smartphone |
| 13 | svetocopy | Svetocopy paper |
| 14 | banknotes10 | 10 ruble banknote |
| 15 | banknotes50 | 50 ruble banknote |
| 16 | banknotes100 | 100 ruble banknote |
| 17 | banknotes200 | 200 ruble banknote |
| 18 | banknotes500 | 500 ruble banknote |
| 19 | banknotes1000 | 1000 ruble banknote |
| 20 | banknotes2000 | 2000 ruble banknote |
| 21 | banknotes5000 | 5000 ruble banknote |
| 22 | mouse | Computer mouse |
| 23 | other | Other objects |

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("format37/russian-rubles-banknotes")

# Access splits
train_data = dataset["train"]
test_data = dataset["test"]

# Example: Get first sample
sample = train_data[0]
image = sample["image"]
objects = sample["objects"]

# Each object has:
# - "bbox": [x, y, width, height] in pixels
# - "category": integer class ID (0-23)
```

## Data Fields

- `image`: PIL Image object
- `width`: Image width in pixels
- `height`: Image height in pixels
- `objects`: List of detected objects, each containing:
  - `bbox`: Bounding box in COCO format `[x, y, width, height]`
  - `category`: Integer class ID (0-23)

## Dataset Statistics

| Split | Images |
|-------|--------|
| Train | {train_count} |
| Test | {test_count} |
| **Total** | **{total_count}** |

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{russian_rubles_banknotes,
  title={{Russian Rubles Banknotes Object Detection Dataset}},
  author={{format37}},
  year={{2024}},
  url={{https://huggingface.co/datasets/format37/russian-rubles-banknotes}}
}}
```

## License

This dataset is released under the MIT License.
'''.format(
        total_count=train_count + test_count,
        train_count=train_count,
        test_count=test_count,
    )

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"Created dataset README at {readme_path}")


def main():
    # Paths
    source_dir = Path("/home/alex/projects/banknotes/banknotes")
    output_dir = Path("/home/alex/projects/banknotes/banknotes-hf-dataset")

    # Clean output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Convert splits
    print("Converting dataset to Hugging Face format...")
    train_count = convert_split(source_dir, output_dir, "train")
    test_count = convert_split(source_dir, output_dir, "test")

    # Create README
    create_dataset_readme(output_dir, train_count, test_count)

    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {train_count + test_count}")


if __name__ == "__main__":
    main()
