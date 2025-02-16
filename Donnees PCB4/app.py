import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

@st.cache_resource
def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
mse_loss = nn.MSELoss()

st.sidebar.title("Paramètres")
st.sidebar.write("Ajustez le seuil pour juger si l'erreur de reconstruction est acceptable ou non.")
threshold = st.sidebar.slider(
    "Seuil d'anomalie",
    min_value=0.0100,
    max_value=0.0200,
    value=0.0115,  
    step=0.0001
)


st.title("EcoDetect")

st.markdown("""
Avec notre approche basée sur les auto-encodeurs convolutifs, si l erreur de reconstruction d une image dépasse le seuil, 
nous la considérons comme anormale. Dans le cas contraire, elle est jugée normale.
""")

uploaded_file = st.file_uploader("Testez votre image (jpg, jpeg, png)", type=["jpg","jpeg","png"])

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image chargée", use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    error = mse_loss(output_tensor, input_tensor).item()
    
    predicted_label = "Anormale" if error > threshold else "Normale"
    
    st.write(f"**Erreur de reconstruction :** {error:.5f}")
    st.write(f"**Seuil actuel :** {threshold:.5f}")
    
    if predicted_label == "Anormale":
        st.error(f"Résultat : {predicted_label}")
    else:
        st.success(f"Résultat : {predicted_label}")
