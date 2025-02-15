import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from codecarbon import EmissionsTracker


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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier de modèle {model_path} n'existe pas.")
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def compute_reconstruction_error(model, img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"L'image {img_path} n'existe pas.")
    
    # Charger et transformer l'image
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0)  # Ajoute une dimension pour le batch
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement ou de la transformation de l'image {img_path}: {e}")
    
    # Passer l'image dans le modèle et calculer l'erreur
    try:
        with torch.no_grad():
            output_tensor = model(input_tensor)
        mse_loss = nn.MSELoss()
        error = mse_loss(output_tensor, input_tensor).item()
    except Exception as e:
        raise RuntimeError(f"Erreur lors du passage de l'image dans le modèle ou du calcul de l'erreur: {e}")

    return error

def detect_anomaly(model, img_path, threshold):
    error = compute_reconstruction_error(model, img_path)
    print(f"Image: {img_path}, Error: {error:.4f}")
    if error > threshold:
        print("=> Anomalie détectée")
        return True
    else:
        print("=> Image normale")
        return False

if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    categories = ["pcb1", "pcb2", "pcb3", "pcb4", "capsules", "macaroni1", "chewinggum", "macaroni2", "candle", "cashew", "pipe_fryum", "fryum"]
    base_path = "C:\\Users\\mouss\\OneDrive\\Documents\\GitHub\\Green-AIHack\\Dataset"
    threshold = 0.0065

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for category in categories:
        model_path = os.path.join(f"{category}_autoencoder.pth")
        model = load_model(model_path)
        if category in ["pcb1", "pcb2", "pcb3", "pcb4", "macaroni1", "macaroni2", "candle"]:
            normal_filename = "0000.JPG"
            anomaly_filename = "091.JPG"
        elif category in ["capsules", "cashew", "chewinggum", "fryum", "pipe_fryum"]:
            normal_filename = "000.jpg"
            anomaly_filename = "091.jpg"
        else:
            normal_filename = "000.jpg"
            anomaly_filename = "0001.jpg"
        
        normal_image_path = os.path.join(base_path, f"{category}\\Data\\Images\\Normal\\{normal_filename}")
        anomaly_image_path = os.path.join(base_path, f"{category}\\Data\\Images\\Anomaly\\{anomaly_filename}")

        # Test image normale
        if detect_anomaly(model, normal_image_path, threshold):
            false_positives += 1
        else:
            true_negatives += 1

        # Test image anormale
        if detect_anomaly(model, anomaly_image_path, threshold):
            true_positives += 1
        else:
            false_negatives += 1

    emissions = tracker.stop()
    print(f"Émissions totales (détection d'anomalies) : {emissions:.6f} kg CO₂ eq")

    # Calcul de l'accuracy
    total_images = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total_images
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
