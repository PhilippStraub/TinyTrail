import asyncio
import logging
from pathlib import Path
import zipfile
import aiofiles
from nats.aio.client import Client as NATS

# Config
NATS_URL = "nats://192.168.188.41:4222"
ANALYSIS_STREAM_NAME = "ANALYSIS"
ANALYSIS_SUBJECT_PREFIX = "analysis.logs"
OUTPUT_DIR = Path("analytical-data-received")
MAX_MSG_SIZE = 8 * 1024 * 1024  # 8 MB
DURABLE_NAME = "analytical-receiver"

# Logging Scheme
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Saves the received ZIP file and extracts it to the corresponding client folder
async def save_and_extract_zip(device_id: str, data: bytes) -> None:
    client_dir = OUTPUT_DIR / device_id
    client_dir.mkdir(parents=True, exist_ok=True)

    zip_path = client_dir / "received.zip"
    async with aiofiles.open(zip_path, "wb") as f:
        await f.write(data)
    logging.info("ZIP saved: %s", zip_path)

    extract_dir = client_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logging.info("Files extracted to: %s", extract_dir)

#Receives messages and processes the ZIP archive
async def message_handler(msg):
    
    device_id = msg.subject.split(".")[-1]
    logging.info("Received message from '%s' (%.2f KB, %d bytes).", device_id, len(msg.data) / (1024), len(msg.data))
    try:
        await save_and_extract_zip(device_id, msg.data)
    except Exception as e:
        logging.exception("Error processing %s: %s", device_id, e)

async def main() -> None:
    nc = NATS()
    await nc.connect(NATS_URL)
    js = nc.jetstream()

    # Ensures Stream exists
    try:
        await js.stream_info(ANALYSIS_STREAM_NAME)
    except Exception:
        logging.error("Stream '%s' does not exist – needs to be created by transmitter first.", ANALYSIS_STREAM_NAME)
        return

    # Stream subscription (continuously)
    await js.subscribe(
        f"{ANALYSIS_SUBJECT_PREFIX}.*",
        durable=DURABLE_NAME,
        cb=message_handler,
        manual_ack=False
    )

    logging.info("Analytical receiver is running waiting for messages...")
    # Will continue until the script is stopped with "STRG + C"
    await asyncio.Event().wait()

def run_main():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Analytical receiver is terminated...")

if __name__ == "__main__":
    run_main()
