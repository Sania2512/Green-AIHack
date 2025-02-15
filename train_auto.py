import os
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
        
        # Decodeur
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
    

def train_model_for_category(category, data_root, batch_size, num_epochs, learning_rate, image_size, model_save_path):
    # Vérifier que le répertoire contient des sous-dossiers
    if not os.path.exists(data_root) or not os.listdir(data_root):
        raise FileNotFoundError(f"Le répertoire {data_root} ne contient pas de sous-dossiers de classes d'images.")

    # Transformations sur les images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Chargement dataset
    train_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ConvAutoencoder()

    # Optimiseur et loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement
    for epoch in range(num_epochs):
        for images, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
        
        print(f"Catégorie: {category}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), model_save_path)
    print(f"Modèle de la catégorie {category} sauvegardé avec succès !")


def main():
    categories = ["pcb1", "pcb2", "pcb3", "pcb4", "capsules", "macaroni1", "chewinggum", "macaroni2", "candle", "cashew", "pipe_fryum", "fryum"]
    base_data_root = "C:\\Users\\mouss\\OneDrive\\Documents\\GitHub\\Green-AIHack\\Dataset"
    batch_size = 32
    num_epochs = 7
    learning_rate = 1e-3
    image_size = (128, 128)  

    for category in categories:
        data_root = os.path.join(base_data_root, f"{category}\\Data\\Images\\Normal")
        model_save_path = f"{category}_autoencoder.pth"
        train_model_for_category(category, data_root, batch_size, num_epochs, learning_rate, image_size, model_save_path)


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()
    main()
    emissions = tracker.stop()
    print(f"Émissions totales (entraînement) : {emissions:.6f} kg CO₂ eq")
