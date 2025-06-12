import albumentations as A


def get_training_augmentation():
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.RandomShadow(p=0.3),
            A.RandomFog(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        ]
    )
