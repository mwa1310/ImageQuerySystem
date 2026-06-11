import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Preprocessing standard pour ResNet et CLIP
transform_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# Normalisation tenseur (1, 3, 224, 224)
def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform_standard(img).unsqueeze(0)

# Chargement d'image en PIL RGB pour CLIP.
def load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def build_image_list(root_dir: str, classes: list) -> list:
    records = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            records.append({"path": os.path.join(cls_dir, fname), "label": cls})
    return records