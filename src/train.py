import torch
import torch.nn as nn  
from torch.utils.data import random_split
from torch.utils.data import DataLoader  
from src.datasets.mvtec import MVTecDataset
from src.models.autoencoder import ConvAutoEncoder
from torchvision import transforms

def train(root, category, epochs=20, batch_size=32, lr=1e-3):
    # 1. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. dataset → random_split 80/20
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # 3. DataLoader for train + val
    full_dataset = MVTecDataset(
        root=root,
        category=category,
        split='train',
        transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # 4. model → to(device)
    model = ConvAutoEncoder().to(device)

    # 5. criterion = MSELoss
    criterion = nn.MSELoss()

    # 6. optimizer = Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. epoch loop:
    #    - train loop: forward → loss → backward → step
    #    - val loop: no gradient → forward → loss only
    #    - save best model if val_loss improves
    best_val_loss = float('inf')
    os.makedirs('weights', exist_ok=True)

    for epoch in range(epochs):
        
        # --- TRAIN LOOP ---
        model.train()
        train_loss = 0.0
        for batch_imgs in train_loader:
            # Handle MVTecDataset targets if it returns (image, label) tuples
            if isinstance(batch_imgs, (list, tuple)):
                batch_imgs = batch_imgs[0]
                
            batch_imgs = batch_imgs.to(device)
            
            # Forward pass
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_imgs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_imgs.size(0)
            
        # --- VAL LOOP ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_imgs in val_loader:
                if isinstance(batch_imgs, (list, tuple)):
                    batch_imgs = batch_imgs[0]
                    
                batch_imgs = batch_imgs.to(device)
                
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_imgs)
                
                val_loss += loss.item() * batch_imgs.size(0)
        
        # Calculate average metric per sample
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")
        
        # --- SAVE BEST MODEL ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = f"weights/best_{category}_autoencoder.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best weights to {save_path}")

    print("Training run finished.")

if __name__ == '__main__':
    train(root='data/mvtec', category='bottle')