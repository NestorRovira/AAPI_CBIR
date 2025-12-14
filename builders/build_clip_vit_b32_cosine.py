import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image

import torch
import faiss
from transformers import CLIPModel, CLIPProcessor

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_CSV = REPO_ROOT / "indices" / "db.csv"

OUT_DIR = REPO_ROOT / "indices" / "clip_vit_b32_cosine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "index.faiss"
EMB_PATH = OUT_DIR / "embeddings.npy"

MODEL_ID = "openai/clip-vit-base-patch32"  # ViT-B/32
BATCH_SIZE = 32
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

    print(f"[CLIP] Loading model: {MODEL_ID} on {DEVICE}")
    model = CLIPModel.from_pretrained(
        MODEL_ID,
        use_safetensors=True
    ).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # CLIP image embedding dim (normalmente 512)
    dim = model.config.projection_dim

    paths = load_db_paths(DB_CSV)
    n = len(paths)
    print(f"[CLIP] Images in db.csv: {n} | Embedding dim: {dim}")

    embs = np.zeros((n, dim), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch_paths = paths[i : i + BATCH_SIZE]
            images = [Image.open(resolve_path(p)).convert("RGB") for p in batch_paths]

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            feats = model.get_image_features(**inputs)  # (B, dim)
            feats = feats.detach().cpu().numpy().astype(np.float32)

            embs[i : i + len(batch_paths)] = feats

    print(f"[CLIP] Embeddings computed in {time.time() - t0:.2f}s")

    faiss.normalize_L2(embs)
    np.save(EMB_PATH, embs)
    print(f"[CLIP] Saved embeddings: {EMB_PATH}")

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[CLIP] Saved FAISS index: {INDEX_PATH} | ntotal={index.ntotal}")


if __name__ == "__main__":
    main()
