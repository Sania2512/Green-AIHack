import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from codecarbon import EmissionsTracker
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class MacaroniDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.labels = []
        
        for label, category in enumerate(['Normal', 'Anomaly']):
            category_dir = os.path.join(image_dir, category)
            if os.path.exists(category_dir):
                for img_name in os.listdir(category_dir):
                    self.image_paths.append(os.path.join(category_dir, img_name))
                    self.labels.append(label)
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.resnet(x).logits
        return self.classifier(features).squeeze(1)

class MacaroniClassifier:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CustomResNet().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, train_dir, val_dir, batch_size=32, epochs=5, learning_rate=0.001):
        train_dataset = Pbc4Dataset(train_dir, self.transform)
        val_dataset = Pbc4Dataset(val_dir, self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_auc = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            self.model.eval()
            val_predictions, val_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_auc = roc_auc_score(val_labels, val_predictions)
            print(f"Validation AUC-ROC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def predict(self, image_path, threshold=0.5):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image).item()
            return {'is_abnormal': output > threshold, 'abnormal_probability': output}
        


def main():
    classifier = Pbc4Classifier()
    train_dir = os.path.join("data", "pbc4", "Data", "Images")
    val_dir = os.path.join("val", "pbc4")
    classifier.train(train_dir, val_dir, batch_size=32, epochs=5, learning_rate=0.001)
    
    test_image = "0090.jpg"
    result = classifier.predict(test_image)
    print(f"Anormal: {result['is_abnormal']}, Probabilité: {result['abnormal_probability']:.4f}")

if __name__ == "__main__":
    try:
        tracker = EmissionsTracker()
        tracker.start()
        main()
        emissions = tracker.stop()
        print(f"Émissions totales (entraînement) : {emissions:.6f} kg CO₂ eq")
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")
