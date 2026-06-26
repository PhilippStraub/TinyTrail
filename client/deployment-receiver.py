import os
import requests
import zipfile
import io
import subprocess
import uuid
import shutil
import time
import logging
import statistics
from collections import deque
from multiprocessing import Process
import asyncio
from pathlib import Path
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from nats.errors import TimeoutError, NoRespondersError
from nats.js.api import ConsumerConfig, AckPolicy
from transmitter import LogTransmitter

# Log Config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Nats Config
SERVER_URL = "http://your-server/register"
BASE_DIR = Path.home() / "tinysphere-deployment-pipeline"
DEPLOYMENTS_DIR = BASE_DIR / "deployments"
CONFIG_DIR = BASE_DIR / "config"
DELETE_OLD = True
NATS_URL = "nats://192.168.178.62:4222"  # JetStream Broker
STREAM_NAME = "MODELS"
MAX_MSG_SIZE = 8 * 1024 * 1024  # 8 MB
SUBJECT_PREFIX = "models.update"


# Networking monitor to collect metrics regarind deployment: Latency, etc.
# This checks whether NATS is down
class NetworkMonitor:
    def __init__(self, max_samples=50, downtime_threshold=30):
        self.latencies = deque(maxlen=max_samples)
        self.last_seq = None
        self.lost_packets = 0
        self.first_packet_ts = None
        self.last_packet_ts = None
        self.last_msg_time = time.time()
        self.downtime_threshold = downtime_threshold

    def record_message(self, msg, start_time, end_time):
        latency_ms = (end_time - start_time) * 1000
        self.latencies.append(latency_ms)
        self.last_msg_time = end_time

        if self.first_packet_ts is None:
            self.first_packet_ts = start_time
        self.last_packet_ts = end_time

        # Checks whether packages got lost
        try:
            seq = msg.metadata.sequence.stream
        except AttributeError:
            seq = None

        if seq is not None:
            if self.last_seq is not None and seq != self.last_seq + 1:
                lost = seq - self.last_seq - 1
                self.lost_packets += max(0, lost)
                logging.warning(
                    f"Packet loss detected: {lost} packets missing (seq jump {self.last_seq} → {seq})"
                )
            self.last_seq = seq


        logging.info(
            f"Latency={latency_ms:.2f} ms | Lost={self.lost_packets} | "
            f"FirstPkt={time.strftime('%H:%M:%S', time.localtime(self.first_packet_ts))} | "
            f"LastPkt={time.strftime('%H:%M:%S', time.localtime(self.last_packet_ts))}"
        )

    def check_downtime(self):
        now = time.time()
        if now - self.last_msg_time > self.downtime_threshold:
            logging.error(f"Network downtime detected (> {self.downtime_threshold}s without messages)")
            self.last_msg_time = now  # reset timer to avoid repeated alerts


monitor = NetworkMonitor()


# Deployment Manager processes incoming deployments
# If a deployment is currently executed, it gets terminated and files are deleted
# New deployment is automatically executed
class DeploymentManager:
    def __init__(self, log_transmitter):
        self.current_process = None
        self.current_path = None
        self.log_transmitter = log_transmitter

    async def stop_current(self):
        if self.current_path:
            await self.log_transmitter.transmit_logs(self.current_path)

        if self.current_process and self.current_process.is_alive():
            logging.info("Terminating current model...")
            self.current_process.terminate()
            self.current_process.join(timeout=5)
            logging.info("Terminated.")

        if DELETE_OLD and self.current_path and os.path.exists(self.current_path):
            logging.info(f"Deleting old deployment: {self.current_path}")
            shutil.rmtree(self.current_path, ignore_errors=True)

        self.current_process = None
        self.current_path = None

    # Searches for a main.py to execute from the received deployment
    # If there is only one .py file, not called main, it will execute it
    # Otherwise an error is logged
    def start_new(self, deployment_path):
        py_files = [f for f in os.listdir(deployment_path) if f.endswith('.py')]

        if not py_files:
            logging.warning("No .py files found in deployment.")
            return

        script_to_run = None
        if len(py_files) == 1:
            script_to_run = py_files[0]
        else:
            main_candidates = [f for f in py_files if "main" in f.lower()]
            if len(main_candidates) == 1:
                script_to_run = main_candidates[0]
            else:
                logging.error("Multiple .py files found and no unique 'main' match.")
                logging.error(f"Files: {py_files}")
                return

        script_path = os.path.join(deployment_path, script_to_run)
        logging.info(f"Selected script to run: {script_path}")

        def run_model():
            logging.info(f"Running model: {script_path}")
            os.chdir(deployment_path)
            os.execvp("python3", ["python3", script_path])

        process = Process(target=run_model)
        process.start()

        self.current_process = process
        self.current_path = deployment_path


# Searches for a device ID within file system (must be provided beforehand)
def initialize():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    device_id_file = CONFIG_DIR / "device_id.txt"
    try:
        if device_id_file.exists():
            device_id = device_id_file.read_text().strip()
            logging.info(f"Found existing device ID: {device_id}")
        else:
            logging.error(f"No device ID found, please provide one in {device_id_file}")
            return None
        return device_id
    except Exception as e:
        logging.error(f"Error retrieving device ID: {e}")
        return None


# Handles NATS Jetstream messages
async def message_handler(msg, deployment_manager):
    start_time = time.time()
    logging.info("New deployment received via NATS JetStream.")
    await deployment_manager.stop_current()

    zip_bytes = io.BytesIO(msg.data)
    deployment_name = f"deployment_{uuid.uuid4().hex[:8]}"
    deployment_path = DEPLOYMENTS_DIR / deployment_name
    os.makedirs(deployment_path, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            safe_extract(zip_ref, deployment_path)
    except Exception as e:
        logging.error(f"Error during unpacking: {e} - message will not be acknowledged.")
        return 

    # Here deployment manager is called to execute the newly received deployment
    deployment_manager.start_new(str(deployment_path))

    # Once the deployment has been successfully processed (ZIP), ack is send to NATS manually
    await msg.ack()
    end_time = time.time()
    monitor.record_message(msg, start_time, end_time)

# Saves received deployments 
def safe_extract(zip_ref, target_dir):
    members = zip_ref.namelist()
    common_prefix = os.path.commonprefix(members)
    if common_prefix and common_prefix.endswith('/'):
        top_level = {m.split('/')[0] for m in members if '/' in m}
        if len(top_level) == 1:
            for member in members:
                new_name = member[len(common_prefix):]
                if not new_name:
                    continue
                source = zip_ref.open(member)
                target_path = os.path.join(target_dir, new_name)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(source.read())
            return
    zip_ref.extractall(target_dir)


# NATS Jetstream Connection
async def start_consumer(device_id, deployment_manager):
    nc = NATS()
    log_transmitter = LogTransmitter(device_id, nc)
    deployment_manager.log_transmitter = log_transmitter
    disconnect_time = None  # Time of disconnect

    async def disconnected_cb():
        nonlocal disconnect_time
        disconnect_time = time.time()
        logging.error("Disconnected from NATS → Network downtime")

    async def reconnected_cb():
        nonlocal disconnect_time
        if disconnect_time:
            downtime = time.time() - disconnect_time
            logging.info(f"Reconnected to NATS (downtime {downtime:.2f} s)")
            disconnect_time = None
        else:
            logging.info("Reconnected to NATS")

    async def closed_cb():
        logging.error("Connection to NATS closed → Downtime until restart")

    await nc.connect(
        NATS_URL,
        disconnected_cb=disconnected_cb,
        reconnected_cb=reconnected_cb,
        closed_cb=closed_cb,
    )

    # Initialize transmitter and ensure stream exists
    await log_transmitter.ensure_stream_exists()

    js = nc.jetstream()
    subject = f"{SUBJECT_PREFIX}.{device_id}"

    try:
        streams = await js.streams_info()
    except TimeoutError as e:
        logging.error(f"Timeout while fetching streams: {e}")
        await nc.close()
        return
    except NoRespondersError as e:
        logging.error(f"No response while fetching streams: {e}")
        await nc.close()
        return
    except Exception as e:
        logging.error(f"Unexpected error while fetching streams: {e}")
        await nc.close()
        return

    stream_names = [s.config.name for s in streams]

    if STREAM_NAME not in stream_names:
        logging.info(f"Stream '{STREAM_NAME}' does not exist. Creating...")
        config = StreamConfig(
            name=STREAM_NAME,
            subjects=[f"{SUBJECT_PREFIX}.*"],
            storage="file",
            retention="workqueue",
            max_msg_size=MAX_MSG_SIZE
        )
        try:
            await js.add_stream(config)
            logging.info(f"Stream '{STREAM_NAME}' created successfully.")
        except Exception as e:
            logging.error(f"Error creating stream: {e}")
            await nc.close()
            return
    else:
        logging.info(f"Stream '{STREAM_NAME}' exists.")

    durable_name = f"{device_id}_worker"
    consumer_config = ConsumerConfig(
        durable_name=durable_name,
        ack_policy=AckPolicy.EXPLICIT,
        ack_wait=60,
        max_deliver=0
    )

    await js.subscribe(
        subject,
        durable=durable_name,
        config=consumer_config,
        cb=lambda msg: asyncio.create_task(message_handler(msg, deployment_manager))
    )

    logging.info(f"Listening for deployments on subject '{subject}'...")

    # Start periodic log transmission
    async def periodic_log_transmission():
        while True:
            await asyncio.sleep(300)
            logging.info("Performing scheduled log transmission...")
            await log_transmitter.transmit_logs(deployment_manager.current_path)

    # Watch for natural process completion and transmit logs immediately
    async def watch_process_completion():
        while True:
            await asyncio.sleep(10)
            proc = deployment_manager.current_process
            path = deployment_manager.current_path
            if proc is not None and not proc.is_alive() and path is not None:
                logging.info("[ProcessWatcher] Inference process finished. Transmitting logs immediately...")
                deployment_manager.current_process = None
                deployment_manager.current_path = None
                await log_transmitter.transmit_logs(path)
                logging.info("[ProcessWatcher] Log transmission complete.")

    transmission_task = asyncio.create_task(periodic_log_transmission())
    completion_watcher_task = asyncio.create_task(watch_process_completion())

    try:
        while True:
            await asyncio.sleep(3600) 
    except asyncio.CancelledError:
        logging.info("Consumer task cancelled.")
    finally:     
        logging.info("Draining NATS connection...")
        await nc.drain()
        
        logging.info("Closing NATS connection...")
        await nc.close()



if __name__ == "__main__":
    DEPLOYMENTS_DIR.mkdir(parents=True, exist_ok=True)
    device_id = initialize()
    if not device_id:
        exit(1)

    deployment_manager = DeploymentManager(log_transmitter=None)

    existing_deployments = sorted(DEPLOYMENTS_DIR.iterdir(), key=os.path.getmtime, reverse=True)
    if existing_deployments:
        latest_deployment = existing_deployments[0]
        logging.info(f"Found existing deployment: {latest_deployment}. Starting it...")
        deployment_manager.start_new(str(latest_deployment))
    else:
        logging.info("No existing deployment found locally.")

    try:
        asyncio.run(start_consumer(device_id, deployment_manager))
    except KeyboardInterrupt:
        logging.info("Interrupted. Cleaning up...")
        # Cleanup is done in the 'finally' block of start_consumer

