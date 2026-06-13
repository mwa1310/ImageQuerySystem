# Image Query System - Moteur de Recherche Visuel

## Présentation

Ce projet implémente un moteur de recherche d'images par similarité visuelle.
À partir d'une image requête, le système retrouve les K images visuellement
les plus similaires dans une base de données de vêtements.

---

## Dataset

**Clothing Dataset Small** - [alexeygrigorev/clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small)

| Split      | Images |
|------------|--------|
| Train      | 3068   |
| Validation | 341    |
| Test       | 372    |
| **Total**  | **3781** |

10 classes : dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt.

---

## Architecture

```
ImageQuerySystem/
├── data/   # Dataset et embeddings (gitignored)
│   ├── clothing-dataset/   # Cloné depuis GitHub
│   └── embeddings/   # Embeddings CLIP + index FAISS
├── notebook/
│   └── Image_QSyst.ipynb   # Notebook principal (4 phases)
├── src/
│   ├── encoders.py   # ResNet-50, CLIP, Autoencoder
│   ├── indexer.py   # Index FAISS
│   ├── evaluate.py   # Precision@K, Recall@K
│   ├── utils.py   # Preprocessing images
│   └── app.py   # Interface Gradio
├── results/   # Graphiques et métriques
├── models/   # Poids autoencoder (gitignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Méthodologie

### Phase 1 - Analyse exploratoire
- Distribution des classes, vérification intégrité, preprocessing
- Resize 224×224, normalisation ImageNet
- Split natif conservé : 81% / 9% / 10%

### Phase 2 - Encodeurs d'images
Trois approches comparées via t-SNE :

| Encodeur    | Dimensions | Remarque                          |
|-------------|------------|-----------------------------------|
| ResNet-50   | 2048       | Pré-entraîné ImageNet             |
| CLIP        | 768        | Meilleure séparation inter-classe |
| Autoencoder | 50176      | Entraîné sur le dataset           |

**CLIP retenu** comme encodeur principal.

### Phase 3 - Index FAISS
- Embeddings CLIP générés pour 3068 images
- Index FlatL2 avec normalisation L2 (similarité cosinus)
- Fonction de requête Top-K

### Phase 4 - Évaluation et interface

| Métrique    | K=1    | K=5    | K=10   |
|-------------|--------|--------|--------|
| Precision@K | 84.68% | 82.2%  | 80.56% |
| Recall@K    | 0.32%  | 1.55%  | 3.01%  |

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/mwa1310/ImageQuerySystem.git
cd ImageQuerySystem

# 2. Créer et activer l'environnement virtuel
python -m venv env
env\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Cloner le dataset
git clone https://github.com/alexeygrigorev/clothing-dataset-small.git data/clothing-dataset
```

---

## Utilisation

### Notebook
Ouvre `notebook/Image_QSyst.ipynb` et exécute les cellules dans l'ordre.

### Interface Gradio
```bash
python src/app.py 
```
Ouvre `http://127.0.0.1:7860` dans le navigateur.

---

## Auteur
MEZAGO Wilfried Aymar - Ecole Nationale Supérieure Polytechnique de Yaoundé 1 - 2025/2026.
