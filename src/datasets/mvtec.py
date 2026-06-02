# src/datasets/mvtec.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch

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
        self.dataset_path = os.path.join(root, category)
        
        self._load_dataset()  # YOU implement this
        print(f"Loaded {len(self.image_paths)} images from path: {os.path.abspath(self.dataset_path)}")

    def _load_dataset(self):
        split_dir = os.path.join(self.dataset_path, self.split)
        
        # Check if the split directory actually exists on disk
        if not os.path.exists(split_dir):
            print(f"Warning: Directory does not exist: {split_dir}")
            return

        for root, dirs, files in os.walk(split_dir):
            for file in files:
                # Use lower() on the file extension to catch .PNG, .JPG, etc.
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                
                # Check for train split safely using lowercase normalization
                if self.split.lower() == 'train':
                    if folder_name.lower() != 'good':
                        continue
                    is_anomaly = 0
                else:
                    is_anomaly = 0 if folder_name.lower() == 'good' else 1
                
                if is_anomaly == 0:
                    mask_path = None
                else:
                    # FIX: Add [0] index to get string name out of splitext tuple
                    base_name = os.path.splitext(file)[0]
                    mask_file = f"{base_name}_mask.png"
                    mask_path = os.path.join(
                        self.dataset_path, 'ground_truth', folder_name, mask_file
                    )
                    if not os.path.exists(mask_path):
                        mask_path = None
                
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.labels.append(is_anomaly)


    
        # print(self.image_paths, self.mask_paths, self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
         
        # 1. Load image and convert to RGB
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        # 2. Apply main image transformations
        if self.transform:
            image = self.transform(image)
            
        # 3. Train mode logic: return image only
        if self.split == 'train':
            return image
            
        # 4. Test mode logic
        if label == 0 or self.mask_paths[idx] is None:
            # Check if image is already a tensor (from transform) or still a PIL Image
            if isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]
            else:
                width, height = image.size
            # Create a matching single-channel blank mask tensor
            mask = torch.zeros((1, height, width), dtype=torch.float32)
        else:
            # Load actual grayscale mask
            mask = Image.open(self.mask_paths[idx]).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            elif isinstance(image, torch.Tensor):
                # If image is a tensor but mask isn't transformed, auto-convert mask to tensor
                
                mask = ToTensor()(mask)
                
        return image, label, mask

