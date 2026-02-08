#!/usr/bin/env python3
"""
Analyze class distribution in the banknote dataset.
Shows which classes are underrepresented and suggests merging strategies.
"""

import json
from collections import defaultdict
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()

# Class ID to name mapping
ID_TO_CLASS_NAME = {
    0: "check",
    1: "napkins",
    2: "glue",
    3: "banknotes",
    4: "banknotes_batch",
    5: "calculator",
    6: "bcard",
    7: "partpack",
    8: "photo_paper",
    9: "iceberg_card",
    10: "document",
    11: "package",
    12: "smartphone",
    13: "svetocopy",
    14: "banknotes10",
    15: "banknotes50",
    16: "banknotes100",
    17: "banknotes200",
    18: "banknotes500",
    19: "banknotes1000",
    20: "banknotes2000",
    21: "banknotes5000",
    22: "mouse",
    23: "other",
}

def analyze_dataset():
    """Analyze class distribution in train and test sets."""

    # Load datasets
    hf_token = os.getenv('HF_TOKEN')
    train_dataset = load_dataset(
        "format37/russian-rubles-banknotes",
        split='train',
        token=hf_token
    )
    test_dataset = load_dataset(
        "format37/russian-rubles-banknotes",
        split='test',
        token=hf_token
    )

    # Count objects per class
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    train_images_with_class = defaultdict(int)
    test_images_with_class = defaultdict(int)

    for item in train_dataset:
        classes_in_image = set()
        for obj in item['objects']:
            cat_id = obj['category']
            train_counts[cat_id] += 1
            classes_in_image.add(cat_id)
        for cat_id in classes_in_image:
            train_images_with_class[cat_id] += 1

    for item in test_dataset:
        classes_in_image = set()
        for obj in item['objects']:
            cat_id = obj['category']
            test_counts[cat_id] += 1
            classes_in_image.add(cat_id)
        for cat_id in classes_in_image:
            test_images_with_class[cat_id] += 1

    # Get all classes present
    all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

    print("=" * 100)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 100)
    print()
    print(f"Total train images: {len(train_dataset)}")
    print(f"Total test images: {len(test_dataset)}")
    print(f"Total classes: {len(all_classes)}")
    print()

    # Print detailed table
    print(f"{'ID':<4} {'Class Name':<20} {'Train Obj':<10} {'Train Img':<10} {'Test Obj':<10} {'Test Img':<10} {'Total Obj':<10} {'Status':<15}")
    print("-" * 100)

    very_small = []
    small = []
    medium = []
    large = []

    for cat_id in all_classes:
        name = ID_TO_CLASS_NAME.get(cat_id, f"class_{cat_id}")
        train_obj = train_counts.get(cat_id, 0)
        test_obj = test_counts.get(cat_id, 0)
        train_img = train_images_with_class.get(cat_id, 0)
        test_img = test_images_with_class.get(cat_id, 0)
        total_obj = train_obj + test_obj

        # Categorize
        if total_obj <= 5:
            status = "⚠️ VERY SMALL"
            very_small.append((cat_id, name, total_obj))
        elif total_obj <= 20:
            status = "⚠️ SMALL"
            small.append((cat_id, name, total_obj))
        elif total_obj <= 100:
            status = "✓ MEDIUM"
            medium.append((cat_id, name, total_obj))
        else:
            status = "✓ LARGE"
            large.append((cat_id, name, total_obj))

        print(f"{cat_id:<4} {name:<20} {train_obj:<10} {train_img:<10} {test_obj:<10} {test_img:<10} {total_obj:<10} {status:<15}")

    print()
    print("=" * 100)
    print("SUMMARY BY SIZE")
    print("=" * 100)
    print()

    print(f"VERY SMALL (≤5 objects): {len(very_small)} classes")
    for cat_id, name, count in very_small:
        print(f"  - {name} (ID {cat_id}): {count} objects")

    print()
    print(f"SMALL (6-20 objects): {len(small)} classes")
    for cat_id, name, count in small:
        print(f"  - {name} (ID {cat_id}): {count} objects")

    print()
    print(f"MEDIUM (21-100 objects): {len(medium)} classes")
    for cat_id, name, count in medium:
        print(f"  - {name} (ID {cat_id}): {count} objects")

    print()
    print(f"LARGE (>100 objects): {len(large)} classes")
    for cat_id, name, count in large:
        print(f"  - {name} (ID {cat_id}): {count} objects")

    print()
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()

    # Analyze banknote denomination classes
    banknote_denoms = {
        14: "banknotes10",
        15: "banknotes50",
        16: "banknotes100",
        17: "banknotes200",
        18: "banknotes500",
        19: "banknotes1000",
        20: "banknotes2000",
        21: "banknotes5000",
    }

    print("BANKNOTE DENOMINATIONS:")
    for cat_id, name in banknote_denoms.items():
        if cat_id in all_classes:
            total = train_counts.get(cat_id, 0) + test_counts.get(cat_id, 0)
            print(f"  - {name}: {total} objects")

    print()
    print("STRATEGY 1: Remove very small classes (≤5 objects)")
    print("  Pros: Cleaner dataset, better class balance")
    print("  Cons: Lose some diversity")
    to_remove = [name for _, name, _ in very_small]
    print(f"  Classes to remove: {', '.join(to_remove)}")

    print()
    print("STRATEGY 2: Merge generic banknote classes")
    print("  - Merge 'banknotes' + 'banknotes_batch' → 'banknotes_generic'")
    print("  - Keep specific denominations (10, 50, 100, etc.) separate")
    print("  Pros: Focus on denomination detection")
    print("  Cons: Lose batch vs single distinction")

    print()
    print("STRATEGY 3: Merge background/distractor classes")
    print("  - Merge 'check', 'napkins', 'glue', 'partpack', etc. → 'background'")
    print("  - Keep only: banknotes, documents, cards, phone, calculator")
    print("  Pros: Focus on important objects, reduce class imbalance")
    print("  Cons: Lose fine-grained classification")

    print()
    print("STRATEGY 4: Keep only banknote-related classes")
    print("  - Keep: All banknote denominations + banknotes + banknotes_batch")
    print("  - Remove: All non-banknote objects")
    print("  Pros: Pure banknote detector, focused task")
    print("  Cons: Can't detect context objects")

    return {
        'very_small': very_small,
        'small': small,
        'medium': medium,
        'large': large,
        'train_counts': train_counts,
        'test_counts': test_counts
    }

if __name__ == "__main__":
    analyze_dataset()
