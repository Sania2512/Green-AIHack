import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Partie encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        # Partie décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Pour avoir une sortie dans [0,1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    # --- 1. Paramètres de configuration ---
    data_root = "macaroni1\Data\Images\Normal"  # Chemin vers vos images normales
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    image_size = (128, 128)  # Adapter si besoin
    model_save_path = "autoencoder.pth"

    # --- 2. Transformations sur les images ---
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # --- 3. Chargement du dataset ---
    # Assurez-vous que data_root contient un sous-dossier (ex: "normal/") avec vos images
    train_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 4. Instanciation du modèle ---
    model = ConvAutoencoder()
    # Si vous avez un GPU, décommentez la ligne suivante :
    # model.cuda()

    # --- 5. Définition de la fonction de perte et de l’optimiseur ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 6. Boucle d’entraînement ---
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            # Si GPU, décommentez :
            # images = images.cuda()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Affichage de la perte moyenne sur l’époque
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # --- 7. Sauvegarde du modèle entraîné ---
    torch.save(model.state_dict(), model_save_path)
    print(f"Modèle sauvegardé sous : {model_save_path}")

if __name__ == "__main__":
    main()
