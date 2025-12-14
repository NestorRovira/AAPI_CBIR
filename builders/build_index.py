import os

# Workaround OpenMP (si lo necesitas en Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import csv

import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights

import faiss

# Workaround OpenMP (si lo necesitas en Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_CSV = REPO_ROOT / "indices" / "db.csv"

# Elige extractor aquí: "resnet50" o "mobilenet_v3"
EXTRACTOR = "mobilenet_v3"

OUT_DIR = REPO_ROOT / "indices" / EXTRACTOR
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "index.faiss"
EMB_PATH = OUT_DIR / "embeddings.npy"   # opcional

IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# PREPROCESS (ImageNet)
# =========================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# LOAD MODEL (Extractor)
# =========================
def load_extractor(name: str):
    name = name.lower().strip()
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        dim = 2048
        return model, dim
    elif name == "mobilenet_v3":
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier = torch.nn.Identity()  # quita clasificador
        # salida típica 960
        dim = 960
        return model, dim
    else:
        raise ValueError(f"Extractor no soportado: {name}")

# =========================
# READ db.csv (orden fijo)
# =========================
def read_db(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in reader.fieldnames:
            raise ValueError("db.csv debe tener columna 'image_path'")
        for r in reader:
            rows.append(r)
    return rows

# =========================
# EMBEDDING EXTRACTION
# =========================
def extract_all_embeddings(image_paths, model, out_dim):
    model = model.to(DEVICE)
    model.eval()

    embs = np.zeros((len(image_paths), out_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            batch_tensors = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                x = preprocess(img)
                batch_tensors.append(x)

            x_batch = torch.stack(batch_tensors, dim=0).to(DEVICE)   # (B,3,224,224)
            y_batch = model(x_batch)                                  # (B,D)
            y_batch = y_batch.detach().cpu().numpy().astype(np.float32)

            embs[i:i+len(batch_paths)] = y_batch

    return embs

# =========================
# BUILD FAISS INDEX (L2)
# =========================
def build_faiss_index(embs: np.ndarray):
    # FAISS necesita contiguo float32
    xb = np.ascontiguousarray(embs.astype(np.float32))
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index

# =========================
# MAIN
# =========================
def main():
    if not DB_CSV.exists():
        raise FileNotFoundError(f"No existe {DB_CSV}. Primero genera db.csv.")

    rows = read_db(DB_CSV)
    image_paths = [r["image_path"] for r in rows]

    # Validación rápida: existen archivos
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Hay {len(missing)} paths que no existen. Ej: {missing[0]}")

    model, out_dim = load_extractor(EXTRACTOR)
    print(f"Extractor: {EXTRACTOR} | Dimensión: {out_dim} | Imágenes: {len(image_paths)} | Device: {DEVICE}")

    embs = extract_all_embeddings(image_paths, model, out_dim)
    print("Embeddings generados:", embs.shape)

    # (opcional) guardar embeddings
    np.save(EMB_PATH, embs)
    print("Embeddings guardados en:", EMB_PATH)

    index = build_faiss_index(embs)
    print("FAISS index listo. Vectores indexados:", index.ntotal)

    faiss.write_index(index, str(INDEX_PATH))
    print("Index guardado en:", INDEX_PATH)

if __name__ == "__main__":
    main()
