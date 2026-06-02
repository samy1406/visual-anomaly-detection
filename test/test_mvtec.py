# quick_test.py
from torchvision import transforms
from src.datasets.mvtec import MVTecDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

ds_train = MVTecDataset('data/', 'bottle', split='train', transform=transform)
ds_test  = MVTecDataset('data/', 'bottle', split='test',  transform=transform, mask_transform=transform)

print(len(ds_train), len(ds_test))
print(ds_train[0].shape)

img, label, mask = ds_test[0]
print(img.shape, mask.shape, label)


# Find the first anomaly index in the test set
defect_idx = next(i for i, label in enumerate(ds_test.labels) if label == 1)
img, label, mask = ds_test[defect_idx]

print(f"Defect Image: {img.shape} | Label: {label} | Mask: {mask.shape}")
print(f"Mask values range from: {mask.min().item()} to {mask.max().item()}")
