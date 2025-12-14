import os
# Workaround OpenMP (si te daba el error OMP)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ✅ AÑADIDO: FAISS
import faiss



# =========================
# 1) RUTAS (CORREGIDO)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]

# 👉 Ajusta SOLO este subpath según tu estructura real dentro de data/raw
DATASET_DIR = REPO_ROOT / "data" / "raw" / "test"   # cambia "test" por tu carpeta real

print("REPO_ROOT:", REPO_ROOT)
print("DATASET_DIR:", DATASET_DIR)

# =========================
# 2) PREPROCESADO
# =========================
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 3) CARGAR MODELO
# =========================
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()
print("Modelo cargado: ResNet50")

# =========================
# 4) ENCONTRAR IMÁGENES
# =========================
if not DATASET_DIR.exists():
    raw_dir = REPO_ROOT / "data" / "raw"
    msg = f"No existe la ruta: {DATASET_DIR}\n"
    if raw_dir.exists():
        msg += "Contenido de data/raw:\n"
        msg += "\n".join([f" - {p.name}" for p in raw_dir.iterdir()])
    else:
        msg += f"Tampoco existe: {raw_dir} (¿seguro que tienes carpeta data/ en la raíz?)"
    raise FileNotFoundError(msg)

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths += list(DATASET_DIR.rglob(ext))

print("Imágenes encontradas:", len(image_paths))
if len(image_paths) == 0:
    raise SystemExit("No se han encontrado imágenes. Revisa que dentro haya .jpg/.png y subcarpetas.")

# solo 10 para pruebas
image_paths = image_paths[:50]

# =========================
# 5) IMAGEN -> EMBEDDING
# =========================
embeddings = []

with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        emb = model(x).squeeze(0).cpu().numpy()
        embeddings.append(emb)

embeddings = np.vstack(embeddings)

# =========================
# 6) RESULTADOS
# =========================
print("Embeddings shape:", embeddings.shape)  # esperado (10, 2048)
print("Ejemplo vector (primeros 10):", embeddings[0][:10])
print("Ejemplo imagen:", image_paths[0])

# =========================
# 7) ✅ FAISS: INDEXAR + BUSCAR TOP-K
# =========================

# FAISS espera float32 y memoria contigua
db = np.ascontiguousarray(embeddings.astype("float32"))
d = db.shape[1]  # dimensión (2048)

# Índice exacto con distancia L2 (simple para empezar)
index = faiss.IndexFlatL2(d)
index.add(db)  # añade todos los vectores de la "biblioteca"

print("\nFAISS index listo. Vectores indexados:", index.ntotal)

# --- Consulta: usa la primera imagen como query (puedes cambiarla) ---
query_id = 1
q = db[query_id:query_id+1]  # shape (1, d)

k = min(5, index.ntotal)  # top-5 (o menos si hay pocas)
D, I = index.search(q, k)

print(f"\nQUERY: {image_paths[query_id]}")
print("Top-K IDs devueltos por FAISS:", I[0].tolist())
print("Distancias L2:", D[0].tolist())

print("\nResultados (paths):")
for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
    print(f"{rank:02d}) id={idx:3d}  dist={dist:.4f}  -> {image_paths[idx]}")
