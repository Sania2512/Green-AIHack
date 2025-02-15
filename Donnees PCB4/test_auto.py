import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from codecarbon import EmissionsTracker
import numpy as np


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

if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    model = ConvAutoencoder()
    model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()

    # prépare le DataLoader de test
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(
        root = r"dataset_split\\test",
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mse_loss = nn.MSELoss()

    # 1) Récupérer toutes les erreurs et labels
    all_errors = []
    all_labels = []

    for images, labels in test_loader:
        with torch.no_grad():
            outputs = model(images)
        error = mse_loss(outputs, images).item()

        # label = 0 (normal) ou 1 (anomaly)
        all_errors.append(error)
        all_labels.append(labels.item())

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)

    # 2) Faire un petit grid search sur le seuil
    best_threshold = None
    best_accuracy = 0.0

    # Par exemple, balayer entre 0.005 et 0.02 par pas de 0.0005
    thresholds = np.arange(0.005, 0.02, 0.0005)

    for thr in thresholds:
        correct = 0
        total = len(all_errors)
        for i in range(total):
            predicted_label = 1 if all_errors[i] > thr else 0
            if predicted_label == all_labels[i]:
                correct += 1
        accuracy = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thr

    print(f"Best threshold found: {best_threshold:.4f}, accuracy = {best_accuracy:.4f}")

    # 3) Affichage des moyennes (optionnel)
    normal_mask = (all_labels == 0)
    anomaly_mask = (all_labels == 1)

    if np.sum(normal_mask) > 0:
        mean_normal_error = np.mean(all_errors[normal_mask])
    else:
        mean_normal_error = 0

    if np.sum(anomaly_mask) > 0:
        mean_anomaly_error = np.mean(all_errors[anomaly_mask])
    else:
        mean_anomaly_error = 0

    print(f"Mean Normal Error: {mean_normal_error:.4f}")
    print(f"Mean Anomaly Error: {mean_anomaly_error:.4f}")

    emissions = tracker.stop()
    print(f"Émissions totales (test) : {emissions:.6f} kg CO₂ eq")