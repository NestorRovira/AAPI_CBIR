import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import re
import csv
import time
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

import torch
import faiss

from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights,
)

from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, Dinov2Model

# =========================
# CONFIG
# =========================

REPO_ROOT = Path(__file__).resolve().parent.parent

INDICES_DIR = REPO_ROOT / "indices"
DB_CSV = INDICES_DIR / "db.csv"
TEST_DIR = REPO_ROOT / "data" / "raw" / "test"

TOPK = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_REGEX = re.compile(r"^([A-Za-z_]+)")

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DINO_MODEL_ID = "facebook/dinov2-small"

# =========================
# PATHS + LABELS
# =========================
def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)

def extract_label_from_filename(path: Path) -> str:
    m = LABEL_REGEX.match(path.stem)
    if not m:
        return "UNKNOWN"
    return m.group(1).rstrip("_")

def list_test_images(test_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

def load_db(db_csv: Path) -> tuple[list[str], list[str]]:
    if not db_csv.exists():
        raise FileNotFoundError(f"No existe {db_csv}")

    image_paths = []
    labels = []
    with open(db_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key = "image_path" if "image_path" in reader.fieldnames else "image"
        has_label = "label" in reader.fieldnames

        for row in reader:
            p = row[key]
            image_paths.append(p)
            if has_label:
                labels.append(str(row["label"]))
            else:
                labels.append(extract_label_from_filename(resolve_path(p)))

    return image_paths, labels

def list_available_extractors() -> list[str]:
    extractors = []
    for d in INDICES_DIR.iterdir():
        if d.is_dir() and (d / "index.faiss").exists():
            extractors.append(d.name)
    if not extractors:
        raise RuntimeError("No se encontraron índices en /indices")
    return sorted(extractors)

# =========================
# FAISS
# =========================
def load_faiss_index(extractor_name: str) -> faiss.Index:
    idx_path = INDICES_DIR / extractor_name / "index.faiss"
    if not idx_path.exists():
        raise FileNotFoundError(f"No existe {idx_path}")
    return faiss.read_index(str(idx_path))

def is_cosine_index(index) -> bool:
    return getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT

# =========================
# PREPROCESS (torchvision)
# =========================
_preprocess_torchvision = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _base_extractor_name(extractor_name: str) -> str:
    return extractor_name[:-7] if extractor_name.endswith("_cosine") else extractor_name

# =========================
# MODEL LOADERS (cache simple in-memory)
# =========================
_model_cache = {}

def load_torchvision_model(base_name: str):
    key = f"tv::{base_name}"
    if key in _model_cache:
        return _model_cache[key]

    if base_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        dim = 2048
    elif base_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        dim = 960
    elif base_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        dim = 1280
    else:
        raise ValueError(f"Extractor torchvision no soportado: {base_name}")

    model = model.to(DEVICE).eval()
    _model_cache[key] = (model, dim)
    return model, dim

def load_clip():
    key = "clip"
    if key in _model_cache:
        return _model_cache[key]

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        use_safetensors=True
    ).to(DEVICE).eval()

    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    dim = model.config.projection_dim
    _model_cache[key] = (model, proc, dim)
    return model, proc, dim

def load_dinov2():
    key = "dinov2"
    if key in _model_cache:
        return _model_cache[key]

    proc = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = Dinov2Model.from_pretrained(DINO_MODEL_ID).to(DEVICE).eval()
    dim = model.config.hidden_size
    _model_cache[key] = (model, proc, dim)
    return model, proc, dim

# =========================
# EMBEDDING DISPATCH
# =========================
def extract_embedding(img: Image.Image, extractor_name: str) -> np.ndarray:
    if extractor_name.startswith("clip_"):
        model, proc, dim = load_clip()
        inputs = proc(images=[img.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        vec = feats.detach().cpu().numpy().astype(np.float32)
        if vec.shape[1] != dim:
            raise ValueError(f"Dim CLIP mismatch: {vec.shape[1]} vs {dim}")
        return vec

    if extractor_name.startswith("dinov2_"):
        model, proc, dim = load_dinov2()
        inputs = proc(images=[img.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
        vec = cls.detach().cpu().numpy().astype(np.float32)
        if vec.shape[1] != dim:
            raise ValueError(f"Dim DINO mismatch: {vec.shape[1]} vs {dim}")
        return vec

    base = _base_extractor_name(extractor_name)
    model, dim = load_torchvision_model(base)
    x = _preprocess_torchvision(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    vec = y.detach().cpu().numpy().astype(np.float32)
    if vec.shape[1] != dim:
        raise ValueError(f"Dim torchvision mismatch: {vec.shape[1]} vs {dim}")
    return vec

def search_topk(index: faiss.Index, q: np.ndarray, k: int) -> np.ndarray:
    q = np.ascontiguousarray(q, dtype=np.float32)
    if is_cosine_index(index):
        faiss.normalize_L2(q)
    _, I = index.search(q, k)
    return I[0]

# =========================
# METRICS
# =========================
def precision_at_k(retrieved_labels: list[str], true_label: str, k: int) -> float:
    top = retrieved_labels[:k]
    return float(sum(1 for lab in top if lab == true_label)) / float(k)

def hit_at_k(retrieved_labels: list[str], true_label: str, k: int) -> int:
    return 1 if true_label in retrieved_labels[:k] else 0

def rr_at_k(retrieved_labels: list[str], true_label: str, k: int) -> float:
    top = retrieved_labels[:k]
    for i, lab in enumerate(top, start=1):
        if lab == true_label:
            return 1.0 / float(i)
    return 0.0

# =========================
# EVAL
# =========================
def evaluate_extractor(extractor_name: str, test_images: list[Path], image_paths: list[str], train_labels: list[str], k: int = 10) -> dict:
    index = load_faiss_index(extractor_name)

    per_query = []
    by_label = defaultdict(list)
    first_hit_ranks = []

    t0 = time.time()
    for qpath in test_images:
        true_label = extract_label_from_filename(qpath)

        img = Image.open(qpath).convert("RGB")
        qvec = extract_embedding(img, extractor_name)
        ids = search_topk(index, qvec, k)

        retrieved_labels = [train_labels[i] for i in ids]
        p = precision_at_k(retrieved_labels, true_label, k)
        h = hit_at_k(retrieved_labels, true_label, k)
        rr = rr_at_k(retrieved_labels, true_label, k)
        top1 = 1 if retrieved_labels[0] == true_label else 0

        rank1 = None
        for r, lab in enumerate(retrieved_labels, start=1):
            if lab == true_label:
                rank1 = r
                break
        if rank1 is not None:
            first_hit_ranks.append(rank1)

        rec = {
            "query": str(qpath),
            "true_label": true_label,
            "top1_label": retrieved_labels[0],
            "precision@k": p,
            "hit@k": h,
            "rr@k": rr,
            "top1_correct": top1,
            "first_hit_rank": rank1,
        }
        per_query.append(rec)
        by_label[true_label].append(rec)

    elapsed = time.time() - t0

    p_vals = [r["precision@k"] for r in per_query]
    h_vals = [r["hit@k"] for r in per_query]
    rr_vals = [r["rr@k"] for r in per_query]
    top1_vals = [r["top1_correct"] for r in per_query]

    label_summary = {}
    for lab, rows in by_label.items():
        label_summary[lab] = {
            "n": len(rows),
            "precision@k_mean": float(np.mean([x["precision@k"] for x in rows])),
            "hit@k_rate": float(np.mean([x["hit@k"] for x in rows])),
            "top1_acc": float(np.mean([x["top1_correct"] for x in rows])),
            "mrr@k": float(np.mean([x["rr@k"] for x in rows])),
        }

    rank_counts = Counter(first_hit_ranks)

    return {
        "extractor": extractor_name,
        "index_metric": "IP(cosine)" if is_cosine_index(index) else "L2",
        "k": k,
        "n_queries": len(per_query),
        "elapsed_s": elapsed,
        "precision@k_mean": float(np.mean(p_vals)) if p_vals else 0.0,
        "precision@k_std": float(np.std(p_vals)) if p_vals else 0.0,
        "hit@k_rate": float(np.mean(h_vals)) if h_vals else 0.0,
        "top1_acc": float(np.mean(top1_vals)) if top1_vals else 0.0,
        "mrr@k": float(np.mean(rr_vals)) if rr_vals else 0.0,
        "first_hit_rank_counts": dict(sorted(rank_counts.items(), key=lambda x: x[0])),
        "per_label": dict(sorted(label_summary.items(), key=lambda x: x[0])),
    }

def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"No existe {TEST_DIR}")
    if not DB_CSV.exists():
        raise FileNotFoundError(f"No existe {DB_CSV}")

    test_images = list_test_images(TEST_DIR)
    if not test_images:
        raise RuntimeError(f"No hay imágenes en {TEST_DIR}")

    image_paths, train_labels = load_db(DB_CSV)
    extractors = list_available_extractors()

    all_results = []
    for ex in extractors:
        res = evaluate_extractor(ex, test_images, image_paths, train_labels, k=TOPK)
        all_results.append(res)

        print("\n" + "=" * 80)
        print(f"EXTRACTOR: {res['extractor']}")
        print(f"Queries: {res['n_queries']} | Top-K: {res['k']} | Time: {res['elapsed_s']:.2f}s")
        print("-" * 80)
        print(f"Precision@{res['k']}: {res['precision@k_mean']:.4f}  (std {res['precision@k_std']:.4f})")
        print(f"Hit@{res['k']}:       {res['hit@k_rate']:.4f}")
        print(f"Top-1 Acc:     {res['top1_acc']:.4f}")
        print(f"MRR@{res['k']}:       {res['mrr@k']:.4f}")

        print("\nDistribución del rank del primer acierto (si existe):")
        for rank, cnt in res["first_hit_rank_counts"].items():
            print(f"  rank {rank:2d}: {cnt}")

        print("\nPor clase (label) [media sobre queries de esa clase]:")
        for lab, stats in res["per_label"].items():
            print(f"  {lab:20s} n={stats['n']:2d}  "
                  f"P@{res['k']}={stats['precision@k_mean']:.3f}  "
                  f"Hit@{res['k']}={stats['hit@k_rate']:.3f}  "
                  f"Top1={stats['top1_acc']:.3f}  "
                  f"MRR@{res['k']}={stats['mrr@k']:.3f}")

    print("\n" + "=" * 90)
    print("RESUMEN COMPARATIVO (todos los extractores detectados en ./indices)")
    print("-" * 90)
    header = f"{'Extractor':28s}  {'Metric':10s}  {'P@10':>7s}  {'Hit@10':>7s}  {'Top1':>7s}  {'MRR@10':>7s}  {'Time(s)':>8s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['extractor']:28s}  {r['index_metric']:10s}  {r['precision@k_mean']:7.3f}  {r['hit@k_rate']:7.3f}  {r['top1_acc']:7.3f}  {r['mrr@k']:7.3f}  {r['elapsed_s']:8.2f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
