import torch
from src.models.autoencoder import ConvAutoEncoder

model = ConvAutoEncoder()
x = torch.randn(1, 3, 224, 224)  # fake batch of 1
out = model(x)
print(out.shape)  # must be [1, 3, 224, 224]