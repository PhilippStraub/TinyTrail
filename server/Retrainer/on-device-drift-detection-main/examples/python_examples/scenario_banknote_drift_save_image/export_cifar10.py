import tarfile
import pickle
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXTRACTED_DIR = "cifar-10-batches-py"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

OUTPUT_DIR = Path("dataset_cifar10")
DOWNLOAD_DIR = Path("downloads")


def unpickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def save_images(data_dict, split_name: str, start_idx: int = 0):
    data = data_dict[b"data"]
    labels = data_dict[b"labels"]

    # CIFAR-10: N x 3072 -> N x 3 x 32 x 32 -> N x 32 x 32 x 3
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    for i, (img_array, label) in enumerate(zip(data, labels), start=start_idx):
        class_name = CLASS_NAMES[label]
        class_dir = OUTPUT_DIR / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        out_file = class_dir / f"{split_name}_{i:05d}.png"
        Image.fromarray(img_array.astype(np.uint8)).save(out_file)


def main():
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    archive_path = DOWNLOAD_DIR / ARCHIVE_NAME

    if not archive_path.exists():
        print("Lade CIFAR-10 herunter ...")
        urlretrieve(URL, archive_path)

    extracted_path = DOWNLOAD_DIR / EXTRACTED_DIR
    if not extracted_path.exists():
        print("Entpacke Archiv ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)

    print("Exportiere Trainingsdaten ...")
    train_idx = 0
    for batch_no in range(1, 6):
        batch = unpickle(extracted_path / f"data_batch_{batch_no}")
        save_images(batch, "train", start_idx=train_idx)
        train_idx += len(batch[b"labels"])

    print("Exportiere Testdaten ...")
    test_batch = unpickle(extracted_path / "test_batch")
    save_images(test_batch, "test", start_idx=0)

    print(f"Fertig. Export nach: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()