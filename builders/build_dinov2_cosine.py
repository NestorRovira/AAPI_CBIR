import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image

import torch
import faiss
from transformers import AutoImageProcessor, Dinov2Model

# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_CSV = REPO_ROOT / "indices" / "db.csv"

OUT_DIR = REPO_ROOT / "indices" / "dinov2_vit_cosine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "index.faiss"
EMB_PATH = OUT_DIR / "embeddings.npy"

MODEL_ID = "facebook/dinov2-small"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_db_paths(db_csv: Path) -> list[str]:
    paths = []
    with open(db_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key = "image_path" if "image_path" in reader.fieldnames else "image"
        for r in reader:
            paths.append(r[key])
    return paths


def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def main():
    if not DB_CSV.exists():
        raise FileNotFoundError(f"No existe {DB_CSV}")

    print(f"[DINOv2] Loading model: {MODEL_ID} on {DEVICE}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Dinov2Model.from_pretrained(MODEL_ID).to(DEVICE).eval()

    dim = model.config.hidden_size  # small=384, base=768, large=1024

    paths = load_db_paths(DB_CSV)
    n = len(paths)
    print(f"[DINOv2] Images in db.csv: {n} | Embedding dim: {dim}")

    embs = np.zeros((n, dim), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch_paths = paths[i : i + BATCH_SIZE]
            images = [Image.open(resolve_path(p)).convert("RGB") for p in batch_paths]

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            out = model(**inputs)  # last_hidden_state: (B, tokens, dim)
            cls = out.last_hidden_state[:, 0, :]  # CLS token (B, dim)
            cls = cls.detach().cpu().numpy().astype(np.float32)

            embs[i : i + len(batch_paths)] = cls

    print(f"[DINOv2] Embeddings computed in {time.time() - t0:.2f}s")

    faiss.normalize_L2(embs)
    np.save(EMB_PATH, embs)
    print(f"[DINOv2] Saved embeddings: {EMB_PATH}")

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[DINOv2] Saved FAISS index: {INDEX_PATH} | ntotal={index.ntotal}")


if __name__ == "__main__":
    main()
