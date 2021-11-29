import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torchvision.transforms import InterpolationMode

default_transform = T.Compose(
    [T.ToTensor(), T.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR)]
)

default_target_transform = T.Compose(
    [T.Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST)]
)


class CocoSegmentation(CocoDetection):
    def __init__(self, *args, **kwargs) -> None:
        if "transform" not in kwargs:
            kwargs["transform"] = default_transform
        if "target_transform" not in kwargs:
            kwargs["target_transform"] = default_target_transform
        super().__init__(*args, **kwargs)

    def _load_target(self, id: int) -> torch.Tensor:
        # Load binary masks
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        masks = [self.coco.annToMask(ann) for ann in anns]
        cats = [ann["category_id"] for ann in anns]

        # Create uint8 mask from binary masks
        t = self.coco.imgs[anns[0]["image_id"]]
        h, w = t["height"], t["width"]
        x = np.zeros(h, w).astype("uint8")
        for mask, c in zip(masks, cats):
            x[mask] = c
        x = torch.from_numpy(x)
        return x
