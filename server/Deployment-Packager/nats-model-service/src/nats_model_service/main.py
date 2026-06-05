import asyncio
import os
import sys
import zipfile
from pathlib import Path
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from nats.errors import TimeoutError, NoRespondersError
from nats.js.api import RetentionPolicy


# Determine project root and folders relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "raw_data"
ZIP_DIR = PROJECT_ROOT / "zipped_data"

# NATS Config
NATS_URL = "nats://localhost:4222"
STREAM_NAME = "MODELS"
MAX_MSG_SIZE = 8 * 1024 * 1024  # 8 MB

# DEVICE Config
DEVICE_ID = "my_device_01"
SUBJECT = f"models.update.{DEVICE_ID}"


async def ensure_stream(js):
    """[INFO] Create the stream if it doesn't exist, and handle errors."""
    try:
        streams = await js.streams_info()
    except TimeoutError as e:
        print(f"[ERROR] Timeout while fetching streams: {e}")
        return
    except NoRespondersError as e:
        print(f"[ERROR] No response from server while fetching streams: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error while fetching streams: {e}")
        return

    stream_names = [s.config.name for s in streams]

    if STREAM_NAME not in stream_names:
        print(f"[INFO] Stream '{STREAM_NAME}' does not exist. Creating...")
        config = StreamConfig(
            name=STREAM_NAME,
            subjects=[f"models.update.*"],  # Cover all device subjects
            storage="file",
            retention="workqueue",
            max_msg_size=MAX_MSG_SIZE
        )
        try:
            await js.add_stream(config)
            print(f"[OK] Stream '{STREAM_NAME}' created successfully.")
        except TimeoutError as e:
            print(f"[ERROR] Timeout while creating the stream: {e}")
        except NoRespondersError as e:
            print(f"[ERROR] No response from server while creating the stream: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error while creating the stream: {e}")
    else:
        print(f"[INFO] Stream '{STREAM_NAME}' already exists.")
        #await js.delete_stream(STREAM_NAME)


async def publish_to_nats(zip_path: Path):
    """Send the ZIP to NATS JetStream"""
    nc = NATS()
    await nc.connect(NATS_URL)
    js = nc.jetstream()

    # Ensure stream exists
    await ensure_stream(js)

    # Publish message
    with open(zip_path, "rb") as f:
        data = f.read()

    ack = await js.publish(SUBJECT, data)
    size_bytes = zip_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    size_mb_str = f"{size_mb:.2f}".replace('.', ',')
    print(f"[OK] ZIP {zip_path.name} published to NATS (seq={ack.seq}) | Size: {size_bytes} bytes = {size_mb_str} MB")

    await nc.close()


def create_zip(folder: Path) -> Path:
    """[INFO] Create a ZIP archive from the given folder and place it in ZIP_DIR"""
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_DIR / f"{folder.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(folder.parent)
                zipf.write(file_path, arcname=rel_path)

    print(f"[OK] ZIP created: {zip_path}")
    return zip_path


async def run(folder_name: str):
    folder = RAW_DIR / folder_name
    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder '{folder}' does not exist.")
        sys.exit(1)

    # Step 1: create ZIP
    zip_path = create_zip(folder)

    # Step 2: send to NATS
    await publish_to_nats(zip_path)


def app():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <folder_name>")
        print()
        print("Example:")
        print("  python main.py scenario_cifar10_knn_benchmark")
        print()
        print("Expected folder:")
        print(f"  {RAW_DIR}/<folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]
    asyncio.run(run(folder_name))


if __name__ == "__main__":
    app()