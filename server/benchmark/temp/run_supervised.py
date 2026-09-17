"""
Supervisor for the multi-day benchmark_pipeline.py sweep against Hans (Raspberry Pi).

Hans is powered through a Voltcraft DSP-3005 lab supply on COM5 (SCPI over serial,
see ../dsp3005_reader.py) and has been observed to go unreachable mid-sweep — both
WiFi drop-outs and at least one silent, trace-free reboot. When a regular (non-baseline)
experiment hits its result_timeout_seconds, benchmark_pipeline.py raises an unhandled
RuntimeError and exits non-zero (see _run_scenario()/abort_on_timeout in
../benchmark_pipeline.py, which this script deliberately does NOT modify).

This script keeps the sweep going unattended:
  - runs benchmark_pipeline.py as a subprocess, tee'ing stdout/stderr to console + log
  - every 30 min, while it's running, pings Hans; if unreachable, terminates the
    subprocess (so COM5 is free) and triggers recovery
  - on any non-zero exit, also triggers recovery
  - recovery = power-cycle Hans via COM5 (OUTP OFF, wait, OUTP ON), wait for it to come
    back up on the network, then restart the pipeline (skip_completed carries it forward)
  - gives up (and alerts) after too many consecutive failed recoveries, or after a total
    restart budget is exhausted, instead of looping forever

Run with the project's venv interpreter, e.g.:
    ..\.venv311\Scripts\python.exe run_supervised.py

Logs to supervisor.log next to this script (server/benchmark/temp/, a local
gitignored scratch dir — not part of the repo).
"""

import signal
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import serial

TEMP_DIR = Path(__file__).resolve().parent
BENCH_DIR = TEMP_DIR.parent
PYTHON_EXE = BENCH_DIR / ".venv311" / "Scripts" / "python.exe"
PIPELINE_SCRIPT = BENCH_DIR / "benchmark_pipeline.py"
LOG_PATH = TEMP_DIR / "supervisor.log"

HANS_HOST = "hans.local"
DSP_PORT = "COM5"
DSP_BAUD = 115200

PING_CHECK_INTERVAL_S = 30 * 60      # how often to check Hans is alive while pipeline runs
PING_RETRIES = 3                     # per check, so a single dropped ping isn't a false alarm
PING_RETRY_DELAY_S = 5
PING_TIMEOUT_MS = 2000

RECOVERY_POWER_OFF_WAIT_S = 30
RECOVERY_POLL_INTERVAL_S = 15
RECOVERY_MAX_WAIT_S = 5 * 60         # empirically ~60-90s to come back; this leaves margin
RECOVERY_SETTLE_S = 30               # extra time for deployment-receiver to reconnect to NATS
MAX_CONSECUTIVE_RECOVERY_FAILURES = 3
MAX_TOTAL_RESTARTS = 30

TERMINATE_GRACE_S = 20               # time to let a CTRL_BREAK-interrupted subprocess exit
                                      # on its own (closes its own COM5 handle) before a hard kill

# A genuine result_timeout_seconds abort can only happen after the pipeline has
# been running for a while (deploy + up to 1800s wait). A non-zero exit within
# this many seconds of starting is a startup/config error (bad JSON, missing
# file, NATS unreachable, ...) that a Hans power-cycle cannot fix -- recovering
# from it anyway would just power-cycle Hans pointlessly and loop until
# MAX_TOTAL_RESTARTS is burned through. Abort immediately instead.
MIN_RUNTIME_FOR_RECOVERY_S = 5 * 60

_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def ping_once(host: str, timeout_ms: int = PING_TIMEOUT_MS) -> bool:
    result = subprocess.run(
        ["ping", "-n", "1", "-w", str(timeout_ms), host],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def is_hans_reachable(retries: int = PING_RETRIES, delay_s: float = PING_RETRY_DELAY_S) -> bool:
    for attempt in range(retries):
        if ping_once(HANS_HOST):
            return True
        if attempt + 1 < retries:
            time.sleep(delay_s)
    return False


def dsp_send(cmd: str) -> None:
    """Open COM5, send one SCPI command, close. Only call while the pipeline
    subprocess is confirmed stopped (it owns COM5 via PowerLogger while running)."""
    ser = serial.Serial(DSP_PORT, DSP_BAUD, timeout=0.3)
    try:
        time.sleep(0.1)
        ser.write((cmd + "\r").encode("ascii"))
        time.sleep(0.2)
    finally:
        ser.close()


def power_cycle_hans() -> bool:
    """Power Hans off then on via the DSP-3005, and wait for it to come back
    online. Returns True on confirmed recovery, False if it gave up waiting."""
    try:
        log("Recovery: opening COM5, sending 'OUTP OFF'")
        dsp_send("OUTP OFF")
    except Exception as e:
        log(f"Recovery: FAILED to send OUTP OFF ({e}) -> treating as failed recovery attempt")
        return False

    log(f"Recovery: power OFF sent, waiting {RECOVERY_POWER_OFF_WAIT_S}s")
    time.sleep(RECOVERY_POWER_OFF_WAIT_S)

    try:
        log("Recovery: opening COM5, sending 'OUTP ON'")
        dsp_send("OUTP ON")
    except Exception as e:
        log(f"Recovery: FAILED to send OUTP ON ({e}) -> treating as failed recovery attempt")
        return False

    log("Recovery: power ON sent, polling ping until Hans responds "
        f"(up to {RECOVERY_MAX_WAIT_S}s)")
    deadline = time.time() + RECOVERY_MAX_WAIT_S
    while time.time() < deadline:
        if ping_once(HANS_HOST):
            log("Recovery: Hans responds to ping again")
            log(f"Recovery: waiting {RECOVERY_SETTLE_S}s extra for deployment-receiver "
                "to reconnect to NATS before the next deploy")
            time.sleep(RECOVERY_SETTLE_S)
            return True
        time.sleep(RECOVERY_POLL_INTERVAL_S)

    log(f"Recovery: Hans did NOT respond within {RECOVERY_MAX_WAIT_S}s")
    return False


def terminate_pipeline(proc: subprocess.Popen) -> None:
    """Stop the pipeline subprocess so COM5 is released. Tries a soft interrupt
    first (lets it run its own cleanup / close its serial handle), falls back to
    a hard kill after TERMINATE_GRACE_S so recovery is never blocked indefinitely."""
    log("Terminating pipeline subprocess: sending CTRL_BREAK")
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    except Exception as e:
        log(f"CTRL_BREAK failed ({e}), falling back to TerminateProcess")
        proc.terminate()

    try:
        proc.wait(timeout=TERMINATE_GRACE_S)
        log("Pipeline subprocess exited after soft interrupt")
        return
    except subprocess.TimeoutExpired:
        pass

    log(f"Pipeline subprocess still alive after {TERMINATE_GRACE_S}s -> killing it")
    proc.kill()
    proc.wait(timeout=15)
    log("Pipeline subprocess killed")


def _stream_output(pipe, log_file) -> None:
    for line in iter(pipe.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        with _log_lock:
            log_file.write(line)
            log_file.flush()
    pipe.close()


def run_pipeline_once():
    """Run benchmark_pipeline.py to completion or until it needs recovery.

    Returns (returncode, reason, elapsed_s) where reason is "exited" (process
    ended on its own; returncode is meaningful), "unreachable" (we terminated
    it ourselves because Hans stopped responding; returncode is None), or
    "interrupted" (Ctrl+C at the supervisor console; returncode is None).
    elapsed_s is wall-clock seconds the subprocess actually ran.
    """
    log(f"Starting pipeline subprocess: {PYTHON_EXE} {PIPELINE_SCRIPT}")
    started_at = time.time()
    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(PIPELINE_SCRIPT)],
        cwd=str(BENCH_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    stream_thread = threading.Thread(
        target=_stream_output, args=(proc.stdout, log_file), daemon=True
    )
    stream_thread.start()

    last_ping_check = time.time()
    try:
        while True:
            try:
                returncode = proc.wait(timeout=5)
                return returncode, "exited", time.time() - started_at
            except subprocess.TimeoutExpired:
                pass

            if time.time() - last_ping_check >= PING_CHECK_INTERVAL_S:
                last_ping_check = time.time()
                log("Periodic health check: pinging Hans")
                if is_hans_reachable():
                    log("Periodic health check: Hans reachable, sweep continues")
                else:
                    log("Periodic health check: Hans NOT reachable after "
                        f"{PING_RETRIES} attempts -> stopping pipeline for recovery")
                    terminate_pipeline(proc)
                    return None, "unreachable", time.time() - started_at
    except KeyboardInterrupt:
        # Ctrl+C at the console raises KeyboardInterrupt here in the supervisor;
        # CREATE_NEW_PROCESS_GROUP means it does NOT propagate to the pipeline
        # subprocess on its own, so without this it would keep running orphaned
        # and holding COM5. Stop it the same way as an "unreachable" recovery.
        log("Ctrl+C received -> stopping pipeline subprocess before exiting")
        terminate_pipeline(proc)
        return None, "interrupted", time.time() - started_at
    finally:
        stream_thread.join(timeout=10)
        log_file.close()


def main() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("Supervisor starting")

    total_restarts = 0

    while True:
        returncode, reason, elapsed_s = run_pipeline_once()

        if reason == "interrupted":
            log("Supervisor stopped by user (Ctrl+C); pipeline subprocess was terminated cleanly")
            return

        if reason == "exited" and returncode == 0:
            log("benchmark_pipeline.py exited with code 0 -> sweep complete, supervisor stopping")
            return

        if reason == "exited" and elapsed_s < MIN_RUNTIME_FOR_RECOVERY_S:
            log(f"ALERT: benchmark_pipeline.py exited with code {returncode} after only "
                f"{elapsed_s:.0f}s -- too fast to be a result_timeout_seconds abort, this "
                "looks like a startup/config error (bad JSON, missing file, NATS unreachable, "
                "...) that a Hans power-cycle would not fix. Aborting supervisor without "
                "recovery -- check the traceback above / in supervisor.log and fix the "
                "underlying issue before restarting.")
            return

        if reason == "exited":
            log(f"benchmark_pipeline.py exited with code {returncode} after {elapsed_s:.0f}s "
                "(likely a result_timeout_seconds abort) -> recovery required")
        else:
            log("Hans went unreachable mid-sweep -> recovery required")

        total_restarts += 1
        if total_restarts > MAX_TOTAL_RESTARTS:
            log(f"ALERT: reached max total restarts ({MAX_TOTAL_RESTARTS}) -> "
                "aborting supervisor, needs manual attention")
            return

        try:
            consecutive_recovery_failures = 0
            while not power_cycle_hans():
                consecutive_recovery_failures += 1
                log(f"Recovery attempt failed ({consecutive_recovery_failures}/"
                    f"{MAX_CONSECUTIVE_RECOVERY_FAILURES} consecutive)")
                if consecutive_recovery_failures >= MAX_CONSECUTIVE_RECOVERY_FAILURES:
                    log("ALERT: too many consecutive recovery failures -> aborting supervisor, "
                        "possible hardware problem, needs manual attention")
                    return
                log("Retrying recovery sequence")
        except KeyboardInterrupt:
            log("Ctrl+C received during recovery -> stopping supervisor "
                "(no pipeline subprocess is running at this point)")
            return

        log(f"Restarting pipeline (restart #{total_restarts}/{MAX_TOTAL_RESTARTS}); "
            "skip_completed will carry forward already-finished experiments")


if __name__ == "__main__":
    main()
