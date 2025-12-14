import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
evaluate_cbir.py

Evalúa CBIR sobre TODAS las imágenes de data/raw/test usando:
- Un único catálogo: indices/db.csv (con image_path y opcionalmente label)
- Un índice FAISS por extractor: indices/<extractor>/index.faiss
- Un extractor (CNN) por nombre de carpeta: resnet50, mobilenet_v3, efficientnet_b0

Métricas (por extractor):
- Precision@10 (media, std, por query y por clase)
- Top-1 accuracy (hit@1)
- Hit@10 (si aparece al menos 1 resultado correcto en top-10)
- MRR@10 (mean reciprocal rank del primer correcto dentro del top-10)
- Distribución de ranks del primer acierto

Requisitos:
- numpy<2 (por FAISS en muchos entornos Windows)
- faiss-cpu instalado
"""

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

# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parent
INDICES_DIR = REPO_ROOT / "indices"
DB_CSV = INDICES_DIR / "db.csv"
TEST_DIR = REPO_ROOT / "data" / "raw" / "test"

TOPK = 10
DEVICE = torch.device("cpu")

# Regla de label: desde el inicio hasta el primer caracter no alfabético (excepto '_')
LABEL_REGEX = re.compile(r"^([A-Za-z_]+)")


# =========================
# HELPERS: paths & labels
# =========================
def resolve_path(p: str) -> Path:
    """Soporta rutas absolutas o relativas al repo."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def extract_label_from_filename(path: Path) -> str:
    """
    Extrae label según tu regla:
    'nombre_apellido (n).jpg' -> 'nombre_apellido'
    'nombre_apellido_n.jpg'   -> 'nombre_apellido_n'? (ojo)
    Con tu especificación previa, el label es el prefijo alfabético+_.
    Ej: homer_simpson_35 -> homer_simpson_
    Para evitar ese '_' final raro, limpiamos '_' finales.
    """
    stem = path.stem
    m = LABEL_REGEX.match(stem)
    if not m:
        return "UNKNOWN"
    label = m.group(1)
    # Limpieza: si acaba en '_' por el patrón nombre_apellido_123, quitamos '_' finales
    label = label.rstrip("_")
    return label


def list_test_images(test_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs)


# =========================
# LOAD DB + TRAIN LABELS
# =========================
def load_db(db_csv: Path) -> tuple[list[str], list[str]]:
    """
    Devuelve:
    - image_paths: lista ordenada de paths (strings) para mapear FAISS id -> imagen
    - labels: lista ordenada de labels (strings) para mapear FAISS id -> label
      Si el CSV no trae label, se deriva del filename.
    """
    if not db_csv.exists():
        raise FileNotFoundError(f"No existe {db_csv}. Genera indices/db.csv primero.")

    image_paths = []
    labels = []

    with open(db_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in reader.fieldnames and "image" not in reader.fieldnames:
            raise ValueError("db.csv debe tener 'image_path' (recomendado) o 'image'.")

        has_label = "label" in reader.fieldnames

        for row in reader:
            p = row.get("image_path") or row.get("image")
            image_paths.append(p)

            if has_label:
                labels.append(str(row["label"]))
            else:
                labels.append(extract_label_from_filename(resolve_path(p)))

    return image_paths, labels


# =========================
# MODELS: registry (fácil añadir nuevos)
# =========================
def _preprocess_default():
    """Preprocess usado al construir tus índices (Resize 224 + Normalize ImageNet)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


MODEL_REGISTRY = {
    "resnet50": {
        "loader": lambda: models.resnet50(weights=ResNet50_Weights.DEFAULT),
        "strip":  lambda m: setattr(m, "fc", torch.nn.Identity()),
        "dim": 2048,
        "preprocess": _preprocess_default(),
    },
    "mobilenet_v3": {
        "loader": lambda: models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT),
        "strip":  lambda m: setattr(m, "classifier", torch.nn.Identity()),
        "dim": 960,
        "preprocess": _preprocess_default(),
    },
    "efficientnet_b0": {
        "loader": lambda: models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
        "strip":  lambda m: setattr(m, "classifier", torch.nn.Identity()),
        "dim": 1280,
        "preprocess": _preprocess_default(),
    },
}


def load_model(extractor_name: str):
    """
    Carga el modelo para un extractor.
    Para añadir uno nuevo: añade entrada a MODEL_REGISTRY.
    """
    if extractor_name not in MODEL_REGISTRY:
        raise ValueError(f"Extractor '{extractor_name}' no está en MODEL_REGISTRY. Añádelo ahí.")

    spec = MODEL_REGISTRY[extractor_name]
    model = spec["loader"]()
    spec["strip"](model)
    model = model.to(DEVICE).eval()
    return model, spec["preprocess"], spec["dim"]


# =========================
# FAISS: load index
# =========================
def load_faiss_index(extractor_name: str) -> faiss.Index:
    idx_path = INDICES_DIR / extractor_name / "index.faiss"
    if not idx_path.exists():
        raise FileNotFoundError(f"No existe {idx_path}. Construye primero el índice de {extractor_name}.")
    return faiss.read_index(str(idx_path))


# =========================
# EMBEDDING + SEARCH
# =========================
def embed_image(img_path: Path, model, preprocess) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # (1,3,224,224)
    with torch.no_grad():
        y = model(x)
    return y.cpu().numpy().astype(np.float32)  # (1,D)


def search_topk(index: faiss.Index, q: np.ndarray, k: int) -> np.ndarray:
    q = np.ascontiguousarray(q, dtype=np.float32)
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
    """Reciprocal Rank del primer acierto dentro del top-k (0 si no hay)."""
    top = retrieved_labels[:k]
    for i, lab in enumerate(top, start=1):
        if lab == true_label:
            return 1.0 / float(i)
    return 0.0


# =========================
# EVAL CORE (modular)
# =========================
def evaluate_extractor(extractor_name: str, test_images: list[Path], image_paths: list[str], train_labels: list[str], k: int = 10) -> dict:
    """
    Evalúa un extractor para todas las imágenes de test.
    Devuelve un dict con métricas y detalle por query.
    """
    model, preprocess, dim = load_model(extractor_name)
    index = load_faiss_index(extractor_name)

    if index.d != dim:
        # Esto suele indicar que el índice no se construyó con este extractor o cambió la arquitectura.
        raise ValueError(f"Dimensión FAISS ({index.d}) != dim del modelo ({dim}) para {extractor_name}.")

    per_query = []
    by_label = defaultdict(list)
    first_hit_ranks = []

    t0 = time.time()

    for qpath in test_images:
        true_label = extract_label_from_filename(qpath)

        qvec = embed_image(qpath, model, preprocess)            # (1,D)
        ids = search_topk(index, qvec, k)                       # (k,)
        retrieved_paths = [image_paths[i] for i in ids]
        retrieved_labels = [train_labels[i] for i in ids]

        p10 = precision_at_k(retrieved_labels, true_label, k)
        h10 = hit_at_k(retrieved_labels, true_label, k)
        rr10 = rr_at_k(retrieved_labels, true_label, k)
        top1 = 1 if retrieved_labels[0] == true_label else 0

        # rank del primer acierto (1..k) o None
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
            "precision@k": p10,
            "hit@k": h10,
            "rr@k": rr10,
            "top1_correct": top1,
            "first_hit_rank": rank1,
            "retrieved_ids": ids.tolist(),
            "retrieved_paths": retrieved_paths,
            "retrieved_labels": retrieved_labels,
        }
        per_query.append(rec)
        by_label[true_label].append(rec)

    elapsed = time.time() - t0

    # Agregados
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
        "per_query": per_query,
    }


# =========================
# REPORTING
# =========================
def print_report(result: dict, show_queries: bool = False):
    k = result["k"]
    print("\n" + "=" * 80)
    print(f"EXTRACTOR: {result['extractor']}")
    print(f"Queries: {result['n_queries']} | Top-K: {k} | Time: {result['elapsed_s']:.2f}s")
    print("-" * 80)
    print(f"Precision@{k}: {result['precision@k_mean']:.4f}  (std {result['precision@k_std']:.4f})")
    print(f"Hit@{k}:       {result['hit@k_rate']:.4f}   (al menos 1 correcto en top-{k})")
    print(f"Top-1 Acc:     {result['top1_acc']:.4f}")
    print(f"MRR@{k}:       {result['mrr@k']:.4f}   (primer correcto dentro del top-{k})")

    print("\nDistribución del rank del primer acierto (si existe):")
    if result["first_hit_rank_counts"]:
        for r, c in result["first_hit_rank_counts"].items():
            print(f"  rank {r:2d}: {c}")
    else:
        print("  (sin aciertos en top-k)")

    print("\nPor clase (label) [media sobre queries de esa clase]:")
    for lab, s in result["per_label"].items():
        print(f"  {lab:20s}  n={s['n']:2d}  P@{k}={s['precision@k_mean']:.3f}  Hit@{k}={s['hit@k_rate']:.3f}  Top1={s['top1_acc']:.3f}  MRR@{k}={s['mrr@k']:.3f}")

    if show_queries:
        print("\nDetalle por query:")
        for r in result["per_query"]:
            qname = Path(r["query"]).name
            fh = r["first_hit_rank"] if r["first_hit_rank"] is not None else "-"
            print(f"  {qname:35s} true={r['true_label']:18s}  top1={r['top1_label']:18s}  P@{k}={r['precision@k']:.2f}  Hit@{k}={r['hit@k']}  RR@{k}={r['rr@k']:.2f}  first_hit={fh}")


# =========================
# MAIN (eval 3 extractores)
# =========================
def main():
    # Validaciones básicas
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"No existe {TEST_DIR}")
    if not DB_CSV.exists():
        raise FileNotFoundError(f"No existe {DB_CSV}")

    test_images = list_test_images(TEST_DIR)
    if not test_images:
        raise RuntimeError(f"No hay imágenes en {TEST_DIR}")

    image_paths, train_labels = load_db(DB_CSV)

    # Evaluar estos 3 (como pediste)
    extractors_to_test = ["resnet50", "mobilenet_v3", "efficientnet_b0"]

    all_results = []
    for ex in extractors_to_test:
        res = evaluate_extractor(
            extractor_name=ex,
            test_images=test_images,
            image_paths=image_paths,
            train_labels=train_labels,
            k=TOPK
        )
        all_results.append(res)
        print_report(res, show_queries=False)

    # Resumen comparativo final (compacto)
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO (más alto = mejor)")
    print("-" * 80)
    header = f"{'Extractor':18s}  {'P@10':>7s}  {'Hit@10':>7s}  {'Top1':>7s}  {'MRR@10':>7s}  {'Time(s)':>8s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['extractor']:18s}  {r['precision@k_mean']:7.3f}  {r['hit@k_rate']:7.3f}  {r['top1_acc']:7.3f}  {r['mrr@k']:7.3f}  {r['elapsed_s']:8.2f}")

    print("=" * 80)
    print("Nota: si quieres añadir un extractor nuevo, crea indices/<nombre>/index.faiss,")
    print("      y añade su loader en MODEL_REGISTRY (dim + cómo quitar clasificador).")


if __name__ == "__main__":
    main()
