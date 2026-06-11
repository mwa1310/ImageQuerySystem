import os
import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from src.utils import load_image, load_image_pil


# Encodeur 1 : ResNet-50 pré-entraîné (ImageNet)

class ResNetEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # On retire la couche de classification, on garde avgpool
        self.model = nn.Sequential(*list(base.children())[:-1])
        self.model.eval().to(self.device)

    # Image en vecteur de dimension 2048
    def encode(self, image_path: str) -> torch.Tensor:
        tensor = load_image(image_path).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding.squeeze().cpu()

    # Encodage split images, retour (N, 2048)
    def encode_batch(self, paths: list) -> torch.Tensor:
        return torch.stack([self.encode(p) for p in paths])
    

# Encodeur 2 : CLIP (OpenAI via Hugging Face)

class CLIPEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    # Image en vecteur de dimension 512
    def encode(self, image_path: str) -> torch.Tensor:
        img = load_image_pil(image_path)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs   = self.model.vision_model(**inputs)
            embedding = outputs.pooler_output
        return embedding.squeeze().cpu()

    # Encodage split images, retour (N, 512)
    def encode_batch(self, paths: list) -> torch.Tensor:
        return torch.stack([self.encode(p) for p in paths])
    

# Encodeur 3 : Autoencoder convolutif

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 14x14
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderEncoder:
    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvAutoencoder().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # Image en vecteur aplati de dimension 256x14x14 = 50176
    def encode(self, image_path: str) -> torch.Tensor:
        tensor = load_image(image_path).to(self.device)
        with torch.no_grad():
            embedding = self.model.encoder(tensor)
        return embedding.squeeze().flatten().cpu()

    # Encodage split images, retour (N, 50176)
    def encode_batch(self, paths: list) -> torch.Tensor:
        return torch.stack([self.encode(p) for p in paths])

    # Entrainement encodeur
    def train_model(self, dataloader, epochs=10, lr=1e-3):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        history   = []
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch)
                loss   = criterion(output, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
        self.model.eval()
        return history