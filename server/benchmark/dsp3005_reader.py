"""
Voltcraft DSP-3005 serial reader / logger.
SCPI protocol over USB-serial (CH340), 115200 baud, 8N1, CR-terminated.

Every reading is appended to a timestamped CSV in the "dsp3005" subfolder
next to this script. The serial link sustains ~30-45 reads/s; --rate paces
the loop to your target (default 8 Hz). Logging runs until Ctrl+C.

V_set/I_set (the configured voltage/current-limit set-points) are read ONCE
per session instead of every row — they're operator-configured and don't
change mid-run, so re-querying them each row only wasted serial round-trips.
Dropping them from 5 to 3 queries/row raises the achievable row rate from
~5-9 Hz to a theoretical ~10-15 Hz within the same "~30-45 reads/s" budget;
8 Hz leaves margin for read_valid_meas() retries.

Usage (run with this folder's .venv python):
    python dsp3005_reader.py [COM3]                # live log at 8 Hz (default)
    python dsp3005_reader.py COM3 --rate 5         # live log at 5 Hz
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
DEFAULT_RATE = 8.0   # readings per second in --watch mode

# Consecutive identical MEAS:VOLT?/MEAS:CURR? readings this many times in a row
# suggest we're polling faster than the device's own internal ADC update rate —
# further speedups would just re-read a stale cached value, not new information.
CEILING_REPEAT_THRESHOLD = 5

# Per-row CSV columns. Power is computed locally (V*I) to save a serial round-trip.
FIELDS = ["timestamp", "V_set", "I_set", "V_out", "I_out", "P_out", "output"]


def query(ser: serial.Serial, cmd: str) -> str:
    ser.reset_input_buffer()
    ser.write((cmd + "\r").encode("ascii"))
    return ser.read_until(b"\r\n").decode("ascii", errors="replace").strip()


def read_setpoints(ser: serial.Serial) -> dict:
    """Configured V/I set-points — read once per session, not per row (see module docstring)."""
    return {"V_set": query(ser, "VOLT?"), "I_set": query(ser, "CURR?")}


def read_meas(ser: serial.Serial, vset: str = "", iset: str = "") -> dict:
    """Per-row live values (3 serial queries); power derived locally.

    V_set/I_set are passed in from a one-time read_setpoints() call rather than
    re-queried here every row.
    """
    v    = query(ser, "MEAS:VOLT?")
    i    = query(ser, "MEAS:CURR?")
    outp = query(ser, "OUTP?")
    try:
        p = f"{float(v) * float(i):.3f}"
    except ValueError:
        p = ""
    return {"V_set": vset, "I_set": iset,
            "V_out": v, "I_out": i, "P_out": p, "output": outp}


def is_valid_meas(vals: dict) -> bool:
    """Return True when the power supply answered with parseable V/I values."""
    try:
        float(vals.get("V_out", ""))
        float(vals.get("I_out", ""))
        float(vals.get("P_out", ""))
    except (TypeError, ValueError):
        return False
    return True


def read_valid_meas(ser: serial.Serial, vset: str = "", iset: str = "",
                    retries: int = 3, delay_s: float = 0.1) -> dict:
    """Read one valid measurement or raise after repeated blank/invalid replies."""
    last_vals = None
    for attempt in range(max(1, retries)):
        vals = read_meas(ser, vset, iset)
        if is_valid_meas(vals):
            return vals
        last_vals = vals
        if attempt + 1 < retries:
            time.sleep(delay_s)
    raise RuntimeError(f"No valid DSP-3005 measurement received on serial link: {last_vals}")


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

    After stop(), `.achieved_rate_hz` reports the actually-sustained sample
    rate and `.ceiling_hit` is True if consecutive identical MEAS readings
    suggested the device's own internal ADC update rate was reached (see
    CEILING_REPEAT_THRESHOLD) — check these when tuning `rate`.
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
        self._pending_vals = None
        self._setpoints = {"V_set": "", "I_set": ""}
        self._start_perf = None
        self._last_meas = None
        self._repeat_count = 0
        self.ceiling_hit = False
        self.achieved_rate_hz = None

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
        try:
            self._ser = serial.Serial(self.port, BAUD, timeout=TIMEOUT)
            time.sleep(0.1)
            try:
                self.device = read_context(self._ser).get("device")
            except Exception:
                self.device = None
            try:
                self._setpoints = read_setpoints(self._ser)
            except Exception:
                self._setpoints = {"V_set": "", "I_set": ""}
            # Fail immediately if the port opens but the supply no longer answers.
            # Otherwise the benchmark would create timestamp-only power CSVs.
            self._pending_vals = read_valid_meas(
                self._ser, self._setpoints["V_set"], self._setpoints["I_set"]
            )
            self._f, self._writer = make_csv_writer(self.csv_path)
            self._stop.clear()
            self._start_perf = time.perf_counter()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
            raise

    def _check_ceiling(self, vals: dict) -> None:
        """Flag if MEAS readings repeat, suggesting the device isn't updating
        as fast as we're polling (see CEILING_REPEAT_THRESHOLD)."""
        key = (vals.get("V_out"), vals.get("I_out"))
        if key == self._last_meas:
            self._repeat_count += 1
            if self._repeat_count >= CEILING_REPEAT_THRESHOLD:
                self.ceiling_hit = True
        else:
            self._repeat_count = 0
        self._last_meas = key

    def _run(self) -> None:
        flush_every = max(1, int(self.rate))
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                if self._pending_vals is not None:
                    vals = self._pending_vals
                    self._pending_vals = None
                else:
                    vals = read_valid_meas(
                        self._ser, self._setpoints["V_set"], self._setpoints["I_set"]
                    )
                self._check_ceiling(vals)
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
        if self._start_perf is not None and self.n_rows > 0:
            elapsed = time.perf_counter() - self._start_perf
            self.achieved_rate_hz = round(self.n_rows / elapsed, 2) if elapsed > 0 else None
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
            setpoints = read_setpoints(ser)

            print(f"DSP-3005 on {port}  |  {ctx['device']}")
            print(f"target {rate:g} Hz  ->  {csv_path}\nCtrl+C to stop\n")
            n = 0
            last_meas = None
            repeat_count = 0
            ceiling_warned = False
            t_start = time.perf_counter()
            t_status = t_start
            flush_every = max(1, int(rate))   # flush ~once per second
            while True:
                t0 = time.perf_counter()
                vals = read_meas(ser, setpoints["V_set"], setpoints["I_set"])
                key = (vals.get("V_out"), vals.get("I_out"))
                if key == last_meas:
                    repeat_count += 1
                    if repeat_count >= CEILING_REPEAT_THRESHOLD and not ceiling_warned:
                        print(f"\n  [warn] {CEILING_REPEAT_THRESHOLD} identical readings in a row — "
                              f"likely at the device's internal update-rate ceiling, "
                              f"polling faster than {rate:g} Hz won't add resolution.")
                        ceiling_warned = True
                else:
                    repeat_count = 0
                last_meas = key
                log_row(writer, vals)
                n += 1
                if n % flush_every == 0:
                    f.flush()
                # lightweight status line (no full-screen clear -> less overhead)
                if t0 - t_status >= 0.5:
                    achieved = n / (t0 - t_start) if t0 > t_start else 0.0
                    print(f"\r  n={n:<7} V={vals['V_out']:>7} V  "
                          f"I={vals['I_out']:>7} A  P={vals['P_out']:>7} W  "
                          f"out={vals['output']:<3}  achieved={achieved:4.1f}Hz ", end="", flush=True)
                    t_status = t0
                # pace to target rate
                dt = time.perf_counter() - t0
                if period - dt > 0:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nstopped.")
    except (ValueError, IndexError):
        print("Bad --rate/--csv argument. Example: --rate 8 --csv out.csv")
    except serial.SerialException as e:
        print(f"Error opening {port}: {e}")
        print("Close the 'Power Supply PC Monitor' app if it's holding the port.")
    finally:
        if f is not None:
            f.flush()
            f.close()


if __name__ == "__main__":
    main()
