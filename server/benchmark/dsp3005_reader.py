"""
Voltcraft DSP-3005 serial reader / logger.
SCPI protocol over USB-serial (CH340), 115200 baud, 8N1, CR-terminated.

Every reading is appended to a timestamped CSV in the "dsp3005" subfolder
next to this script. The serial link sustains ~30-45 reads/s; --rate paces
the loop to your target (default 5 Hz). Logging runs until Ctrl+C.

Usage (run with this folder's .venv python):
    python dsp3005_reader.py [COM3]                # live log at 5 Hz (default)
    python dsp3005_reader.py COM3 --rate 3         # live log at 3 Hz
    python dsp3005_reader.py COM3 --csv out.csv
"""

import serial
import time
import sys
import csv
import os
import threading
from datetime import datetime

PORT = "COM3"
BAUD = 115200
TIMEOUT = 0.3        # serial read timeout (s)
DEFAULT_RATE = 5.0   # readings per second in --watch mode

# Per-row CSV columns. Power is computed locally (V*I) to save a serial round-trip.
FIELDS = ["timestamp", "V_set", "I_set", "V_out", "I_out", "P_out", "output"]


def query(ser: serial.Serial, cmd: str) -> str:
    ser.reset_input_buffer()
    ser.write((cmd + "\r").encode("ascii"))
    return ser.read_until(b"\r\n").decode("ascii", errors="replace").strip()


def read_meas(ser: serial.Serial) -> dict:
    """All per-row values (5 serial queries); power derived locally."""
    vset = query(ser, "VOLT?")
    iset = query(ser, "CURR?")
    v    = query(ser, "MEAS:VOLT?")
    i    = query(ser, "MEAS:CURR?")
    outp = query(ser, "OUTP?")
    try:
        p = f"{float(v) * float(i):.3f}"
    except ValueError:
        p = ""
    return {"V_set": vset, "I_set": iset,
            "V_out": v, "I_out": i, "P_out": p, "output": outp}


def read_context(ser: serial.Serial) -> dict:
    """Device identity, read once per run."""
    return {"device": query(ser, "*IDN?")}


def make_csv_writer(path: str):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        writer.writeheader()
    return f, writer


def log_row(writer, vals: dict) -> None:
    row = dict(vals)
    row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
    writer.writerow(row)


class PowerLogger:
    """Pipeline-controllable DSP-3005 power logger.

    Opens the serial port, polls V/I/P at `rate` Hz in a daemon thread, and
    appends rows (same FIELDS/format as the CLI) to `csv_path`. Call start()
    before an experiment and stop() after — or use as a context manager.

    Failures (port busy, device unplugged) are captured in `.error` instead of
    raised from the thread, so a power glitch never aborts a long sweep.
    """

    def __init__(self, csv_path, port: str = PORT, rate: float = DEFAULT_RATE):
        self.csv_path = str(csv_path)
        self.port = port
        self.rate = rate
        self.period = 1.0 / rate if rate > 0 else 0.0
        self._stop = threading.Event()
        self._thread = None
        self._ser = None
        self._f = None
        self._writer = None
        self.device = None
        self.n_rows = 0
        self.error = None

    def start(self) -> None:
        # Start each experiment with a FRESH file. make_csv_writer() opens in
        # append mode (right for the standalone CLI), so without this a re-run of
        # the same experiment id would concatenate the previous run's readings.
        try:
            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)
        except OSError:
            pass
        # Open serial up-front so a busy/missing port surfaces immediately
        # (and can be handled by the caller) rather than inside the thread.
        self._ser = serial.Serial(self.port, BAUD, timeout=TIMEOUT)
        time.sleep(0.1)
        try:
            self.device = read_context(self._ser).get("device")
        except Exception:
            self.device = None
        self._f, self._writer = make_csv_writer(self.csv_path)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        flush_every = max(1, int(self.rate))
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                vals = read_meas(self._ser)
                log_row(self._writer, vals)
                self.n_rows += 1
                if self.n_rows % flush_every == 0:
                    self._f.flush()
            except Exception as e:  # serial hiccup — record and stop cleanly
                self.error = str(e)
                break
            dt = time.perf_counter() - t0
            if self.period - dt > 0:
                # wait() (not sleep) so stop() interrupts pacing immediately
                self._stop.wait(self.period - dt)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._f is not None:
            try:
                self._f.flush()
                self._f.close()
            except Exception:
                pass
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def main():
    args  = sys.argv[1:]
    port  = next((a for a in args if not a.startswith("-")), PORT)
    rate  = float(args[args.index("--rate") + 1]) if "--rate" in args else DEFAULT_RATE
    period = 1.0 / rate if rate > 0 else 0.0

    if "--csv" in args:
        csv_path = args[args.index("--csv") + 1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "dsp3005")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"dsp3005_log_{stamp}.csv")

    f = None
    try:
        f, writer = make_csv_writer(csv_path)
        with serial.Serial(port, BAUD, timeout=TIMEOUT) as ser:
            time.sleep(0.1)
            ctx = read_context(ser)

            print(f"DSP-3005 on {port}  |  {ctx['device']}")
            print(f"target {rate:g} Hz  ->  {csv_path}\nCtrl+C to stop\n")
            n = 0
            t_status = time.perf_counter()
            flush_every = max(1, int(rate))   # flush ~once per second
            while True:
                t0 = time.perf_counter()
                vals = read_meas(ser)
                log_row(writer, vals)
                n += 1
                if n % flush_every == 0:
                    f.flush()
                # lightweight status line (no full-screen clear -> less overhead)
                if t0 - t_status >= 0.5:
                    print(f"\r  n={n:<7} V={vals['V_out']:>7} V  "
                          f"I={vals['I_out']:>7} A  P={vals['P_out']:>7} W  "
                          f"out={vals['output']:<3} ", end="", flush=True)
                    t_status = t0
                # pace to target rate
                dt = time.perf_counter() - t0
                if period - dt > 0:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nstopped.")
    except (ValueError, IndexError):
        print("Bad --rate/--csv argument. Example: --rate 4 --csv out.csv")
    except serial.SerialException as e:
        print(f"Error opening {port}: {e}")
        print("Close the 'Power Supply PC Monitor' app if it's holding the port.")
    finally:
        if f is not None:
            f.flush()
            f.close()


if __name__ == "__main__":
    main()
