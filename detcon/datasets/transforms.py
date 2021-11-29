import kornia.augmentation as K


class RandomResizedCrop(K.RandomResizedCrop):
    def __init__(self, *args, **kwargs) -> None:
        if kwargs["align_corners"] is None:
            kwargs["align_corners"] = False
        super().__init__(*args, **kwargs)
        self.align_corners = None


default_augs = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5), data_keys=["input", "mask"]
)

default_ssl_augs = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=0.5),
    RandomResizedCrop(
        size=(224, 224), scale=(0.08, 1.0), resample="NEAREST", align_corners=None
    ),
    K.RandomSolarize(thresholds=0.1, p=0.2),
    data_keys=["input", "mask"],
)
