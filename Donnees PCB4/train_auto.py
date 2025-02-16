import os
import shutil
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from codecarbon import EmissionsTracker



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ENTRAINEMENT DE L'AUTOENCODEUR 

def main():
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-3
    image_size = (128, 128)
    model_save_path = "autoencoder.pth"

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_root = r"dataset_split\\train"  
    val_root   = r"dataset_split\\val"   

    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ConvAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        #  Entraînement 
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # validation 
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss_total += loss.item()
        val_loss = val_loss_total / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Sauvegarder le meilleur modèle
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print("Meilleur modèle sauvegardé avec succès !")
    else:
        print("Aucun modèle sauvegardé (pas de validation?).")


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    main()

    emissions = tracker.stop()
    print(f"Émissions totales (entraînement) : {emissions:.6f} kg CO₂ eq")
