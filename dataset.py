import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CombinedVOC(Dataset):
    def __init__(self, base_path, img_list, transform=None):
        self.img_dir = os.path.join(base_path, "JPEGImages")
        self.gt_dir = os.path.join(base_path, "SegmentationClass")
        self.sam_dir = os.path.join(base_path, "SAM_Segmentation_Maps1")
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, f"{name}.jpg")).convert('RGB'))
        
        # Preference Logic: Use Original gt if available, otherwise use SAM maps
        gt_path = os.path.join(self.gt_dir, f"{name}.png")
        sam_path = os.path.join(self.sam_dir, f"{name}.png")
        
        if os.path.exists(gt_path):
            mask = np.array(Image.open(gt_path))
        elif os.path.exists(sam_path):
            mask = np.array(Image.open(sam_path))
        else:
            mask = np.zeros((img.shape, img.shape[1]), dtype=np.uint8)

        # Do NOT convert 255 to 0; CrossEntropy ignore_index handles it.
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
            
        return img, mask.long()

    def __len__(self):
        return len(self.img_list)