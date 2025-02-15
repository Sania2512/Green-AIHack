import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from codecarbon import EmissionsTracker
 

# Importation ConvAutoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_model(model_path):
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() 
    return model

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor()
])

def compute_reconstruction_error(model, img_path):
    img = Image.open(img_path).convert('RGB') # chargement image
    input_tensor = transform(img).unsqueeze(0)  # taille: [1, 3, 128, 128]

    # Passer l'image dans le modèle 
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Calcul de MSE
    mse_loss = nn.MSELoss()
    error = mse_loss(output_tensor, input_tensor).item()

    return error


def detect_anomaly(model, img_path, threshold):
    error = compute_reconstruction_error(model, img_path)
    print(f"Image: {img_path}, Error: {error:.4f}")
    if error > threshold:
        print("=> Anomalie détectée")
    else:
        print("=> Image normale")

if __name__ == "__main__":

    tracker = EmissionsTracker()
    tracker.start()
    model = load_model("autoencoder.pth")

    threshold = 0.00765
    test_images = [
        "pcb4\\Data\\Images\\Normal\\normal\\0000.JPG",
        "pcb4\\Data\\Images\\Anomaly\\091.JPG"
    ]
    
    for img_path in test_images:
        detect_anomaly(model, img_path, threshold)

    emissions = tracker.stop()
    print(f"Émissions totales (entraînement) : {emissions:.6f} kg CO₂ eq")

   

