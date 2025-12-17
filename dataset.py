import os
import numpy as np
import PIL.Image as Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class GolfHoleSegmentationDataset(Dataset):
    IGNORE_INDEX = 255

    def __init__(self, images, masks, train=True):
        # image and mask paths
        self.images = images
        self.masks = masks
        self.train = train

        # sort images and masks by filename to ensure correct pairing
        self.images.sort(key=lambda x: os.path.basename(x))
        self.masks.sort(key=lambda x: os.path.basename(x))

        # ImageNet normalization for ResNet encoders
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        if self.train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    scale=(0.90, 1.10),
                    rotate=(-20, 20),
                    interpolation=1,
                    mask_interpolation=0,
                    fill=0,
                    fill_mask=self.IGNORE_INDEX,
                    p=0.7
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image and mask
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        mask  = np.array(Image.open(self.masks[idx]).convert('L'))

        # apply augmentations
        out = self.aug(image=image, mask=mask)
        augmented_image = out['image']
        augmented_mask  = out['mask']

        return augmented_image, augmented_mask