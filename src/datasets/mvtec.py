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
        self.dataset_path = "data"
        
        self._load_dataset()  # YOU implement this

    def _load_dataset(self):
        # Hint: os.walk or os.listdir through split folder
        # train → only good/ folder
        # test  → good/ + defect folders
        # YOU: derive label from folder name
        # YOU: find corresponding mask path (or handle missing for good/)
        split_dir = os.path.join(self.dataset_path, self.split)
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                # Filter for images (MVTec uses .png)
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                # Full path to the input image
                img_path = os.path.join(root, file)
                
                # 1. Derive label from the folder name
                folder_name = os.path.basename(root)
                
                if self.split == 'train':
                    # Train split only contains the 'good' folder
                    if folder_name != 'good':
                        continue
                    is_anomaly = 0
                else:
                    # Test split contains 'good' and defect folders (e.g., 'broken', 'contamination')
                    is_anomaly = 0 if folder_name == 'good' else 1

                # 2. Find corresponding mask path
                if is_anomaly == 0:
                    mask_path = None  # No mask exists for 'good' images
                else:
                    # MVTec masks mirror the test structure inside the 'ground_truth' folder
                    # Example: test/broken/000.png -> ground_truth/broken/000_mask.png
                    mask_file = f"{os.path.splitext(file)[0]}_mask.png"
                    mask_path = os.path.join(
                        self.dataset_path, 
                        'ground_truth', 
                        folder_name, 
                        mask_file
                    )
                    
                    # Safety check for missing masks
                    if not os.path.exists(mask_path):
                        mask_path = None

                image_paths.append(img_path)
                mask_paths.append(mask_path)
                labels.append(is_anomaly)

        # Store or return the gathered data
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        print(self.image_paths, self.mask_paths, self.labels)

    def __len__(self):
        pass  # YOU

    def __getitem__(self, idx):
        # Load image → apply transform
        # train: return image only
        # test:  return image, label, mask
        # YOU: what do you return as mask when label=0?
        pass

