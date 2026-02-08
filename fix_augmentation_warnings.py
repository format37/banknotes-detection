"""
Updated augmentation code to fix albumentations warnings.
Apply this AFTER current training completes.
"""

import albumentations as A

def get_updated_train_transform():
    """Updated augmentation pipeline with correct albumentations API."""
    return A.Compose([
        # Geometric transforms - use Affine instead of ShiftScaleRotate
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.7),

        # FIXED: Use Affine instead of ShiftScaleRotate
        A.Affine(
            scale=(0.7, 1.3),      # Same as scale_limit=0.3
            translate_percent=0.1,  # Same as shift_limit=0.1
            rotate=(-15, 15),       # Same as rotate_limit=15
            p=0.6
        ),

        A.Perspective(scale=(0.05, 0.15), p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),

        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),

        # Noise and blur
        # FIXED: GaussNoise - removed var_limit, mean parameters
        A.OneOf([
            A.GaussNoise(p=1.0),  # Uses default variance
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # FIXED: CoarseDropout - use new parameter names
        A.CoarseDropout(
            max_holes=3,
            max_height=60,
            max_width=60,
            # fill_value removed - uses default
            p=0.3
        ),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.3,
        min_area=100
    ))


def get_updated_val_transform():
    """Validation transform - FIXED: no bbox_params when transform list is empty."""
    # For validation, we don't apply augmentations
    # But we still need bbox_params for the compose
    return A.Compose([
        # Empty - no augmentations for validation
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels']
    ))


# Print the updated code
print("=" * 80)
print("UPDATED AUGMENTATION CODE (to fix warnings)")
print("=" * 80)
print()
print("Replace in dataset.py _get_default_transform() method:")
print()
print('''
def _get_default_transform(self, is_train: bool) -> A.Compose:
    """Get default augmentation pipeline with fixed API."""
    if is_train:
        return A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=0.1,
                rotate=(-15, 15),
                p=0.6
            ),
            A.Perspective(scale=(0.05, 0.15), p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),

            # Photometric
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),

            # Noise/blur
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),

            # Dropout
            A.CoarseDropout(max_holes=3, max_height=60, max_width=60, p=0.3),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3,
            min_area=100
        ))
    else:
        return A.Compose([], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
''')

print()
print("=" * 80)
print("NOTE: Apply this AFTER your current training completes!")
print("=" * 80)
