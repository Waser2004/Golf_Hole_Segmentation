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

        # ImageNet normalization for ResNet encoders
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        if self.train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.10,
                    rotate_limit=20,
                    border_mode=0, 
                    value=255, 
                    mask_value=self.IGNORE_INDEX, 
                    p=0.7
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianNoise(p=0.2),
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
        augmented_image = out['mask']

        return augmented_image, augmented_image