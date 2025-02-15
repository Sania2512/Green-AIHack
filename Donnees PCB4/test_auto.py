import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

    # accuracy
    threshold = 0.00765
    correct = 0
    total = 0
    mse_loss = nn.MSELoss()

    for images, labels in test_loader:
        # 0 (normal) ou 1 (anomaly)
        with torch.no_grad():
            outputs = model(images)
        error = mse_loss(outputs, images).item()

        predicted_label = 1 if error > threshold else 0
        true_label = labels.item()
        if predicted_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    emissions = tracker.stop()
    print(f"Émissions totales (test) : {emissions:.6f} kg CO₂ eq")
