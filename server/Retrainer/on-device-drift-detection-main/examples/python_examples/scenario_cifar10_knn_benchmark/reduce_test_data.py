from pathlib import Path
import random
import shutil

# =========================
# KONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "input_images"
TARGET_DIR = BASE_DIR / "input_images_reduced_20_percent"

KEEP_RATIO = 0.20   # 20 % behalten = 80 % verringern
SEED = 42
COPY_FILES = True   # True = kopieren, False = verschieben

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# SCRIPT
# =========================
random.seed(SEED)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Quellordner nicht gefunden: {SOURCE_DIR}")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
if not class_dirs:
    raise ValueError(f"Keine Klassenordner in {SOURCE_DIR} gefunden.")

total_original = 0
total_kept = 0

print(f"Quelle: {SOURCE_DIR}")
print(f"Ziel:   {TARGET_DIR}")
print(f"Behalte pro Klasse: {KEEP_RATIO:.0%}")
print("-" * 50)

for class_dir in sorted(class_dirs):
    images = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"[Übersprungen] {class_dir.name}: keine Bilddateien gefunden")
        continue

    total_original += len(images)
    keep_count = max(1, int(len(images) * KEEP_RATIO))
    selected_images = random.sample(images, keep_count)

    target_class_dir = TARGET_DIR / class_dir.name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in selected_images:
        target_path = target_class_dir / img_path.name
        if COPY_FILES:
            shutil.copy2(img_path, target_path)
        else:
            shutil.move(str(img_path), str(target_path))

    total_kept += keep_count
    print(f"{class_dir.name}: {len(images)} -> {keep_count}")

print("-" * 50)
print(f"Gesamt: {total_original} -> {total_kept}")
print("Fertig.")