import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
import faiss


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_CSV = REPO_ROOT / "indices" / "db.csv"
OUT_DIR = REPO_ROOT / "indices" / "mobilenet_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "index.faiss"
EMB_PATH = OUT_DIR / "embeddings.npy"

IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

weights = MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)
model.classifier = torch.nn.Identity()   # -> 960D
model = model.to(DEVICE).eval()

paths = []
with open(DB_CSV, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        paths.append(r["image_path"])

embs = np.zeros((len(paths), 960), dtype=np.float32)

with torch.no_grad():
    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i:i+BATCH_SIZE]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(imgs).to(DEVICE)
        y = model(x).cpu().numpy()
        embs[i:i+len(batch)] = y

np.save(EMB_PATH, embs)

index = faiss.IndexFlatL2(960)
index.add(embs)
faiss.write_index(index, str(INDEX_PATH))

print("MobileNetV3 index creado:", INDEX_PATH)
