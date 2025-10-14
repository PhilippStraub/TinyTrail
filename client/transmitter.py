import os
import zipfile
import logging
import asyncio
from pathlib import Path
import io
from typing import List, Optional, Tuple
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig

# Nats Config
NATS_URL = "nats://192.168.188.41:4223"
ANALYSIS_STREAM_NAME = "ANALYSIS"
ANALYSIS_SUBJECT_PREFIX = "analysis.logs"
MAX_MSG_SIZE = 8 * 1024 * 1024  # 8 MB

# Packages and sends log files from a deployment's 'logs' directory
# Once sent, files are deleted
class LogTransmitter:
    def __init__(self, device_id: str, nc: NATS):
        self.device_id = device_id
        self.nc = nc
        self.js = self.nc.jetstream(timeout=240)

    # Checks if NATS Analysis stream exists
    async def ensure_stream_exists(self):
        try:
            await self.js.stream_info(ANALYSIS_STREAM_NAME)
        except Exception:
            logging.info(f"Stream '{ANALYSIS_STREAM_NAME}' does not exist. Creating...")
            config = StreamConfig(
                name=ANALYSIS_STREAM_NAME,
                subjects=[f"{ANALYSIS_SUBJECT_PREFIX}.*"],
                storage="file",
                retention="workqueue",
                max_msg_size=MAX_MSG_SIZE
            )
            # Retry creation to handle high latency/timeouts gracefully
            for attempt in range(3):
                try:
                    await self.js.add_stream(config)
                    logging.info(f"Stream '{ANALYSIS_STREAM_NAME}' created successfully.")
                    break
                except Exception as e:
                    if attempt < 2:
                        backoff = 2 * (attempt + 1)
                        logging.warning(f"Error creating stream (attempt {attempt+1}/3): {e}. Retrying in {backoff}s..." ) 
                        await asyncio.sleep(backoff)
                    else:
                        logging.error(f"Failed to create stream after retries: {e}")
                        raise

    def _pack_logs(self, batch_files: List[Path]) -> Tuple[Optional[bytes], List[Path]]:
        zip_buffer = io.BytesIO()
        successfully_zipped_files = []
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in batch_files:
                try:
                    zf.write(file_path, file_path.name)
                    successfully_zipped_files.append(file_path)
                except FileNotFoundError:
                    logging.warning(f"File not found during zipping, skipping: {file_path}")
                    continue
        
        zip_data = zip_buffer.getvalue()
        
        # Safety check for max message size
        if len(zip_data) > MAX_MSG_SIZE:
            return None, []

        return zip_data, successfully_zipped_files

    async def transmit_logs(self, deployment_path: str):
        if not deployment_path:
            return

        log_dir = Path(deployment_path) / "logs"
        if not log_dir.is_dir():
            return

        # Create a snapshot of the file list to avoid race conditions
        try:
            log_files = sorted([f for f in log_dir.iterdir() if f.is_file()])
        except FileNotFoundError:
            logging.warning(f"Log directory not found during file listing: {log_dir}")
            return
            
        if not log_files:
            return

        logging.info(f"Preparing logs in {log_dir} for transmission...")

        # Maximum number of files per batch, just gathering the first 50 files is more efficient than calculation of packages size
        MAX_FILES_PER_BATCH = 50

        # Split files into batches
        file_batches = [
            log_files[i:i + MAX_FILES_PER_BATCH]
            for i in range(0, len(log_files), MAX_FILES_PER_BATCH)
        ]

        for i, batch_files in enumerate(file_batches):
            logging.info(f"Processing batch {i+1}/{len(file_batches)} with {len(batch_files)} files...")

            # Runs the blocking zip operation in a separate thread to avoid blocking the event loop
            zip_data, successfully_zipped_files = await asyncio.to_thread(
                self._pack_logs, batch_files
            )

            if not zip_data:
                logging.warning(
                    f"Batch {i+1}/{len(file_batches)} is too large and will be skipped. "
                    f"The number of files per batch should possibly be reduced."
                )
                continue

            # Send ZIP file via NATS
            try:
                subject = f"{ANALYSIS_SUBJECT_PREFIX}.{self.device_id}"
                await self.js.publish(subject, zip_data, timeout=240)  # For Robustness
                logging.info(f"Log archive batch {i+1}/{len(file_batches)} ({len(zip_data)} bytes) sent successfully to '{subject}'.")

                # Delete successfully sent image files
                image_extensions = ['.jpg', '.jpeg', '.png']
                for file_path in successfully_zipped_files:
                    if file_path.suffix.lower() in image_extensions:
                        try:
                            os.remove(file_path)
                            logging.info(f"Image file deleted: {file_path}")
                        except OSError as e:
                            logging.error(f"Error deleting image file {file_path}: {e}")
            # Once one batch fails, reexecution is done to prevent data loss (since files will be deleted permanently after sending)
            except Exception as e:
                logging.error(f"Error sending log archive batch {i+1}: {e}")
                break
