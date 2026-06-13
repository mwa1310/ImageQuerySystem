import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import faiss
import numpy as np
from PIL import Image
from src.encoders import CLIPEncoder
from src.indexer  import FaissIndex

# Chargement de l'encodeur et de l'index
print("Chargement de l'encodeur CLIP...")
encoder = CLIPEncoder()

print("Chargement de l'index FAISS...")
index = FaissIndex(dimension=768)
index.load(
    "data/embeddings/clip_faiss.index",
    "data/embeddings/clip_meta.npy"
)
print("Prêt.")

# Historique en mémoire
historique = []

# Fonction principale
def search_similar(image: Image.Image, k: int = 5):
    # Sauvegarde temporaire pour l'encodeur
    tmp_path = "tmp_query.jpg"
    image.save(tmp_path)

    # Encodage et recherche
    emb = encoder.encode(tmp_path).numpy().reshape(1, -1).astype("float32")
    faiss.normalize_L2(emb)
    results = index.search(emb.flatten(), k=k)
    os.remove(tmp_path)

    # Résultats
    output_images = [Image.open(r["path"].replace("../", "")) for r in results]
    output_labels = [f"{r['label']} (d={r['distance']:.3f})" for r in results]

    # Ajout à l'historique
    historique.append({
        "requete": image.copy(),
        "resultats": list(zip(output_images, output_labels)),
        "k": k
    })

    return list(zip(output_images, output_labels))

# Historique des recherches
def get_historique():
    if not historique:
        return [], "Aucune recherche effectuée."

    rows = []
    for i, h in enumerate(reversed(historique)):
        rows.append([
            f"Recherche #{len(historique) - i}",
            f"K={h['k']}",
            ", ".join([r[1] for r in h["resultats"]])
        ])
    return rows

def clear_historique():
    historique.clear()
    return [], "Historique effacé."

# Interface Gradio
with gr.Blocks(title="Moteur de Recherche Visuel") as demo:
    gr.Markdown("# Moteur de Recherche Visuel - Clothing Dataset")

    with gr.Tabs():

        # Onglet Recherche
        with gr.Tab(" Recherche"):
            gr.Markdown("Charge une image de vêtement et trouve les plus similaires.")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Image requête")
                    k_slider = gr.Slider(1, 10, value=5, step=1,
                                            label="Nombre de résultats (K)")
                    btn = gr.Button("Rechercher", variant="primary")
                with gr.Column():
                    output_gallery = gr.Gallery(label="Résultats",
                                                columns=5, height=400)
            btn.click(fn=search_similar,
                      inputs=[input_image, k_slider],
                      outputs=output_gallery)

        # Onglet Historique
        with gr.Tab(" Historique"):
            gr.Markdown("Historique des recherches effectuées depuis le démarrage.")
            with gr.Row():
                btn_refresh = gr.Button(" Rafraîchir", variant="secondary")
                btn_clear = gr.Button(" Effacer", variant="stop")

            historique_table = gr.Dataframe(
                headers=["Recherche", "K", "Résultats retournés"],
                datatype=["str", "str", "str"],
                label="Historique",
                interactive=False
            )
            status = gr.Textbox(label="Statut", interactive=False)

            btn_refresh.click(fn=get_historique,
                              outputs=historique_table)
            btn_clear.click(fn=clear_historique,
                            outputs=[historique_table, status])

if __name__ == "__main__":
    demo.launch()