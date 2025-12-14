from pathlib import Path
import csv
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "data" / "raw" / "train"
OUTPUT_CSV = REPO_ROOT / "indices" / "db.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

LABEL_REGEX = re.compile(r"^([A-Za-z_]+)")

rows = []

for img_path in sorted(TRAIN_DIR.iterdir()):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue

    filename = img_path.stem  # sin extensión

    match = LABEL_REGEX.match(filename)
    if not match:
        raise ValueError(f"No se pudo extraer label de: {img_path.name}")

    label = match.group(1)

    rows.append({
        "image_path": str(img_path.as_posix()),
        "label": label
    })


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"db.csv generado con {len(rows)} imágenes")
print(f"Ruta: {OUTPUT_CSV}")
