import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pathlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import faiss
import streamlit as st
from streamlit_cropper import st_cropper

from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights
)

# =============================
# CONFIG BÁSICA
# =============================
st.set_page_config(layout="wide")
DEVICE = torch.device("cpu")

# 🔧 MODIFICADO: app.py está en la raíz del repo
REPO_ROOT = pathlib.Path(__file__).resolve().parent

INDICES_DIR = REPO_ROOT / "indices"
DB_CSV = INDICES_DIR / "db.csv"

IMG_SIZE = 224

# =============================
# REGISTRO DE MODELOS
# =============================
MODEL_REGISTRY = {
    "resnet50": {
        "loader": lambda: (
            models.resnet50(weights=ResNet50_Weights.DEFAULT),
            2048
        ),
        "strip_classifier": lambda m: setattr(m, "fc", torch.nn.Identity())
    },
    "mobilenet_v3": {
        "loader": lambda: (
            models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT),
            960
        ),
        "strip_classifier": lambda m: setattr(m, "classifier", torch.nn.Identity())
    },
    "efficientnet_b0": {
        "loader": lambda: (
            models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
            1280
        ),
        "strip_classifier": lambda m: setattr(m, "classifier", torch.nn.Identity())
    }
}

# =============================
# UTILIDADES
# =============================
def resolve_image_path(p: str) -> str:
    return p if os.path.isabs(p) else str(REPO_ROOT / p)


@st.cache_resource
def load_db():
    df = pd.read_csv(DB_CSV)
    if "image_path" not in df.columns:
        raise ValueError("db.csv debe contener columna 'image_path'")
    return list(df["image_path"].values)


@st.cache_resource
def list_available_extractors():
    extractors = []
    for d in INDICES_DIR.iterdir():
        if d.is_dir() and (d / "index.faiss").exists():
            extractors.append(d.name)
    if not extractors:
        raise RuntimeError("No se encontraron extractores en /indices")
    return sorted(extractors)


@st.cache_resource
def load_faiss_index(extractor_name: str):
    return faiss.read_index(str(INDICES_DIR / extractor_name / "index.faiss"))


@st.cache_resource
def load_model(extractor_name: str):
    if extractor_name not in MODEL_REGISTRY:
        raise ValueError(f"Extractor '{extractor_name}' no registrado")

    model, _ = MODEL_REGISTRY[extractor_name]["loader"]()
    MODEL_REGISTRY[extractor_name]["strip_classifier"](model)
    model.eval()
    return model


# =============================
# PREPROCESS
# =============================
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_embedding(img: Image.Image, extractor_name: str) -> np.ndarray:
    model = load_model(extractor_name)
    x = _preprocess(img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        y = model(x)

    return y.cpu().numpy().astype(np.float32)


def retrieve_images(img_query, extractor_name, k):
    index = load_faiss_index(extractor_name)
    q = extract_embedding(img_query, extractor_name)
    q = np.ascontiguousarray(q, dtype=np.float32)
    _, idxs = index.search(q, k)
    return idxs[0]


# =============================
# STREAMLIT APP
# =============================
def main():
    st.title("CBIR IMAGE SEARCH")

    image_list = load_db()
    extractors = list_available_extractors()

    col1, col2 = st.columns(2)

    with col1:
        st.header("QUERY")

        extractor = st.selectbox("Choose extractor", extractors)

        k = st.slider("Number of images to retrieve", 1, 30, 10)

        img_file = st.file_uploader("Upload image", type=["jpg", "png"])

        if img_file:
            img = Image.open(img_file)
            cropped = st_cropper(img, realtime_update=True, box_color="#FF0004")
            st.image(cropped, caption="Query image", width=200)

    with col2:
        st.header("RESULTS")

        if img_file:
            start = time.time()
            ids = retrieve_images(cropped, extractor, k)
            elapsed = time.time() - start
            st.markdown(f"**Retrieved in {elapsed:.3f}s**")

            cols = st.columns(5)
            for i, idx in enumerate(ids):
                p = resolve_image_path(image_list[idx])
                cols[i % 5].image(Image.open(p), use_column_width=True)


if __name__ == "__main__":
    main()
