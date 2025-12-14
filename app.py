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
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights, EfficientNet_B0_Weights
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, Dinov2Model


# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = pathlib.Path(__file__).resolve().parent
INDICES_DIR = REPO_ROOT / "indices"
DB_CSV = INDICES_DIR / "db.csv"

IMG_SIZE = 224  # para torchvision
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DINO_MODEL_ID = "facebook/dinov2-small"

# =============================
# PREPROCESS
# =============================
_preprocess_torchvision = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# UTILIDADES
# =============================
def resolve_image_path(p: str) -> str:
    return p if os.path.isabs(p) else str(REPO_ROOT / p)

@st.cache_resource
def load_db():
    df = pd.read_csv(DB_CSV)
    if "image_path" not in df.columns and "image" not in df.columns:
        raise ValueError("db.csv debe contener columna 'image_path' (recomendado) o 'image'")
    col = "image_path" if "image_path" in df.columns else "image"
    return list(df[col].values)

@st.cache_resource
def list_available_extractors():
    extractors = []
    for d in INDICES_DIR.iterdir():
        if d.is_dir() and (d / "index.faiss").exists():
            extractors.append(d.name)
    if not extractors:
        raise RuntimeError("No se encontraron extractores en /indices (subcarpetas con index.faiss)")
    return sorted(extractors)

@st.cache_resource
def load_faiss_index(extractor_name: str):
    return faiss.read_index(str(INDICES_DIR / extractor_name / "index.faiss"))

def is_cosine_index(index) -> bool:
    return getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT

# =============================
# MODEL LOADERS (cacheados)
# =============================
def _base_extractor_name(extractor_name: str) -> str:
    return extractor_name[:-7] if extractor_name.endswith("_cosine") else extractor_name

@st.cache_resource
def load_torchvision_model(base_name: str):
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
    return model, dim

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        use_safetensors=True
    ).to(DEVICE).eval()

    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    dim = model.config.projection_dim  # normalmente 512
    return model, proc, dim

@st.cache_resource
def load_dinov2():
    proc = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = Dinov2Model.from_pretrained(DINO_MODEL_ID).to(DEVICE).eval()
    dim = model.config.hidden_size  # small=384
    return model, proc, dim

# =============================
# EMBEDDING DISPATCH
# =============================
def extract_embedding(img: Image.Image, extractor_name: str) -> np.ndarray:
    """
    Devuelve vector (1, D) float32 listo para FAISS.
    """
    # CLIP
    if extractor_name.startswith("clip_"):
        model, proc, dim = load_clip()
        inputs = proc(images=[img.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)  # (1,dim)
        vec = feats.detach().cpu().numpy().astype(np.float32)
        if vec.shape[1] != dim:
            raise ValueError(f"Dim CLIP mismatch: {vec.shape[1]} vs {dim}")
        return vec

    # DINOv2
    if extractor_name.startswith("dinov2_"):
        model, proc, dim = load_dinov2()
        inputs = proc(images=[img.convert("RGB")], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]  # (1,dim)
        vec = cls.detach().cpu().numpy().astype(np.float32)
        if vec.shape[1] != dim:
            raise ValueError(f"Dim DINO mismatch: {vec.shape[1]} vs {dim}")
        return vec

    # Torchvision (resnet50, mobilenet_v3, efficientnet_b0 + *_cosine)
    base = _base_extractor_name(extractor_name)
    model, dim = load_torchvision_model(base)
    x = _preprocess_torchvision(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    vec = y.detach().cpu().numpy().astype(np.float32)
    if vec.shape[1] != dim:
        raise ValueError(f"Dim torchvision mismatch: {vec.shape[1]} vs {dim}")
    return vec

def retrieve_images(img_query: Image.Image, extractor_name: str, k: int) -> np.ndarray:
    index = load_faiss_index(extractor_name)
    q = extract_embedding(img_query, extractor_name)
    q = np.ascontiguousarray(q, dtype=np.float32)

    # Si el índice es IP (cosine), normalizamos la query
    if is_cosine_index(index):
        faiss.normalize_L2(q)

    _, I = index.search(q, k)
    return I[0]

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

        img_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
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
                cols[i % 5].image(Image.open(p), width="content")

if __name__ == "__main__":
    main()
