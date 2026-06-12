import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.paths = []
        self.labels = []

    # Construction index
    def build(self, embeddings: np.ndarray, paths: list, labels: list):
        embs = embeddings.astype("float32").copy()
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.paths  = paths
        self.labels = labels
        print(f"Index construit : {self.index.ntotal} vecteurs")

    # Sauvegarde de l'index
    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        np.save(meta_path, {"paths": self.paths, "labels": self.labels})
        print(f"Index sauvegardé : {index_path}")

    # Chargement de l'index
    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        meta = np.load(meta_path, allow_pickle=True).item()
        self.paths  = meta["paths"]
        self.labels = meta["labels"]
        print(f"Index chargé : {self.index.ntotal} vecteurs")

    # Recherhce
    def search(self, query_emb: np.ndarray, k: int = 5) -> list:
        emb = query_emb.reshape(1, -1).astype("float32")
        faiss.normalize_L2(emb)
        distances, indices = self.index.search(emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "path" : self.paths[idx],
                "label" : self.labels[idx],
                "distance": float(dist)
            })
        return results