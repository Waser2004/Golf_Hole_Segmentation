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

        # blend colors
        self.colors = {
            # background and ignore index
            "0": [0, 0, 0],
            "255": [0, 0, 0],

            # golf hole classes
            "1": [50, 205, 50],
            "2": [104, 155, 64],
            "3": [33, 153, 50],
            "4": [20, 101, 33],
            "5": [17, 76, 25],
            "6": [210, 180, 140],
            "7": [240, 230, 140],
            "8": [17, 48, 25],
            "9": [70, 130, 180],
            "10": [255, 255, 255],
            "11": [128, 128, 128],
            "12": [226, 114, 91]
        }

        # ImageNet normalization for ResNet encoders
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        if self.train:
            self.aug = A.Compose([
                A.VerticalFlip(p=0.5),
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
        mask  = np.array(Image.open([mask for mask in self.masks if os.path.basename(mask) == os.path.basename(self.images[idx])][0]).convert('L'))

        # apply augmentations
        out = self.aug(image=image, mask=mask)
        augmented_image = out['image']
        augmented_mask = out['mask']

        return augmented_image, augmented_mask
    
    def blend(self, image, mask, alpha=0.5):
        """Blend image and mask for visualization."""
        mask_rgb = np.zeros_like(image)
        
        for class_idx, color in self.colors.items():
            mask_rgb[mask == int(class_idx)] = np.array(color)
        
        blended = (alpha * image + (1 - alpha) * mask_rgb).astype(np.uint8)
        return blended
