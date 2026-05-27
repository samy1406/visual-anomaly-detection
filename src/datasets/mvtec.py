# src/datasets/mvtec.py

import os
from PIL import Image
from torch.utils.data import Dataset

class MVTecDataset(Dataset):
    def __init__(self, root, category, split='train', transform=None, mask_transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        self._load_dataset()  # YOU implement this

    def _load_dataset(self):
        # Hint: os.walk or os.listdir through split folder
        # train → only good/ folder
        # test  → good/ + defect folders
        # YOU: derive label from folder name
        # YOU: find corresponding mask path (or handle missing for good/)
        pass

    def __len__(self):
        pass  # YOU

    def __getitem__(self, idx):
        # Load image → apply transform
        # train: return image only
        # test:  return image, label, mask
        # YOU: what do you return as mask when label=0?
        pass