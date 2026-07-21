# Implementierungsplan: PC↔Pi-Zeit-Handshake (Clock-Offset-Messung)

**Status:** Implementiert (2026-07-21), bis auf den echten Hardware-Test
(Checkliste Punkt 4) — der erfordert PC+Pi live über NATS und kann nicht aus
dieser Session heraus ausgeführt werden. Alle Code-Änderungen (Pi-Handler,
PC-Messung, Analyse-Korrektur, Doku) sind umgesetzt und mit synthetischen
Fixtures getestet.

**Vorgeschichte (zum Nachlesen, nicht Voraussetzung zum Verstehen):**
[`MESSMETHODIK_ZEITSYNCHRONISATION.md`](MESSMETHODIK_ZEITSYNCHRONISATION.md),
Abschnitt 3, 5 und 6. Kurzfassung des Problems: Die Leistungsmessung
(`power_*.csv`) wird auf der PC-Uhr zeitgestempelt, das Ressourcen-Log
(`resource_usage_*.csv`, CPU/RAM) auf der Pi-Uhr. Das Inferenzfenster wird aus
dem Pi-Log bestimmt und dann auf die PC-Leistungskurve angewendet — das setzt
voraus, dass beide Uhren übereinstimmen. Aktuell gibt es dafür **keinen
Messwert**, nur eine nachträgliche kausale Plausibilitätsschranke
(`analysis/verify_power_coverage.py`). Dieser Plan ersetzt „plausibel" durch
„gemessen".

---

## Ziel

Vor (mindestens) jedem Experiment und jedem Baseline-Block den Uhrenversatz
zwischen PC und Pi aktiv messen (NTP-artiger Round-Trip über NATS), das
Ergebnis in die Metadaten schreiben, und in der Analyse damit die
Inferenzfenster **korrigieren** statt nur zu vertrauen.

## Protokoll (klassisches NTP-Prinzip, per NATS Request/Reply)

```
PC:  t1 = time.time()                         (PC-Uhr, unmittelbar vor dem Senden)
PC:  → NATS-Request an Subject "time.sync.<device_id>"
Pi:  Empfängt Request, sofort:
Pi:  t2 = time.time()                         (Pi-Uhr, im Handler, keine Verzögerung)
Pi:  → Reply mit {"pi_time": t2}
PC:  t3 = time.time()                         (PC-Uhr, unmittelbar bei Empfang der Antwort)

round_trip_s     = t3 - t1
offset_s         = t2 - (t1 + t3) / 2         (Pi-Uhr minus PC-Uhr, NTP-Formel)
```

Mehrere Pings (empfohlen: n=7) pro Messung; verwendet wird die Probe mit dem
**kleinsten `round_trip_s`** (NTP-Konvention — die schnellste Antwort hat die
symmetrischste, am wenigsten verzerrte Netzwerklaufzeit). Zusätzlich Median
und Std. aller `offset_s`-Proben als Unsicherheitsangabe protokollieren.

**Vorzeichen-Konvention (wichtig, konsistent halten):** `offset_s = Pi-Uhr −
PC-Uhr`. Um einen Pi-Zeitstempel `T_pi` in „PC-äquivalente Zeit" umzurechnen:
`T_pc_equiv = T_pi − offset_s`.

## Warum NATS Request/Reply (nicht JetStream)

Die bestehende Deployment-Pipeline nutzt JetStream (Streams/Consumer,
`ConsumerConfig`, `AckPolicy`) für den Modell-Transfer — das ist für einen
Zeit-Ping unnötig schwergewichtig und würde durch Persistenz/Redelivery-Logik
zusätzliche, variable Latenz einführen (schlecht für eine Zeitmessung, die
gerade auf symmetrischer, minimaler Latenz beruht). Stattdessen: **einfaches
Core-NATS Request/Reply** (`nc.request()` / `msg.respond()`), das im selben
`nc`-Client neben dem bestehenden JetStream-Kram parallel läuft.

---

## Änderungen im Detail

### 1. Pi-Seite: `client/deployment-receiver.py`

Aktuelle Struktur (verifiziert in dieser Session): `start_consumer()`
(ab Zeile 304) verbindet `nc = NATS()`, `await nc.connect(NATS_URL, ...)`,
holt `js = nc.jetstream()` und abonniert den Deployment-Stream über
`js.subscribe(...)` (Zeile 394). Direkt daneben eine **zusätzliche**, simple
Subscription ergänzen:

```python
async def handle_time_sync(msg):
    """Reply immediately with this device's current clock time — no queueing,
    no awaits before capturing pi_time, so the reply latency stays minimal and
    symmetric (required for the NTP-style offset formula to hold)."""
    pi_time = time.time()
    await msg.respond(json.dumps({"pi_time": pi_time}).encode())

# in start_consumer(), neben der bestehenden js.subscribe(...)-Zeile:
await nc.subscribe(f"time.sync.{device_id}", cb=handle_time_sync)
```

`json` muss importiert werden, falls noch nicht vorhanden (kurz prüfen).
`device_id` ist in `start_consumer(device_id, deployment_manager)` bereits
als Parameter vorhanden — Subject-Namenskonvention passend zu
`models.update.<device_id>` / `analysis.logs.<device_id>`.

**Kritisch für die Messgenauigkeit:** `handle_time_sync` darf **nichts**
anderes tun als sofort antworten — keine Locks, kein Logging vor der
Zeitstempel-Erfassung, keine Queue. Jede zusätzliche Verzögerung hier
verzerrt die Offset-Schätzung asymmetrisch (PC→Pi-Latenz ≠ Pi→PC-Latenz).

### 2. PC-Seite: `server/benchmark/benchmark_pipeline.py`

Neue Funktion, in der Nähe von `_subscribe_deploy_collect` (ab Zeile 888)
platzieren:

```python
async def _measure_clock_offset(nc, device_id: str, n_pings: int = 7) -> Dict:
    """NTP-style round-trip clock offset measurement against the Pi's
    time.sync.<device_id> responder (client/deployment-receiver.py). Returns
    the min-round-trip sample plus median/std across all pings as an
    uncertainty indicator. offset_s = Pi clock - PC clock (see PLAN doc for
    sign convention and correction formula)."""
    samples = []
    for _ in range(n_pings):
        t1 = time.time()
        try:
            reply = await nc.request(f"time.sync.{device_id}", b"", timeout=2.0)
            t3 = time.time()
            pi_time = json.loads(reply.data.decode())["pi_time"]
            round_trip = t3 - t1
            offset = pi_time - (t1 + t3) / 2.0
            samples.append({"t1": t1, "t3": t3, "pi_time": pi_time,
                            "round_trip_s": round_trip, "offset_s": offset})
        except Exception as e:
            samples.append({"error": str(e)})
    valid = [s for s in samples if "offset_s" in s]
    if not valid:
        return {"offset_s": None, "error": "no valid ping succeeded", "samples": samples}
    best = min(valid, key=lambda s: s["round_trip_s"])
    offsets = [s["offset_s"] for s in valid]
    return {
        "offset_s": best["offset_s"],
        "round_trip_s": best["round_trip_s"],
        "offset_median_s": float(np.median(offsets)),
        "offset_std_s": float(np.std(offsets, ddof=1)) if len(offsets) > 1 else None,
        "n_pings": n_pings,
        "n_valid": len(valid),
        "measured_at": datetime.now().isoformat(),
        "samples": samples,
    }
```

**Aufruf-Punkte** (jeweils mit derselben, schon offenen `nc`-Verbindung, oder
einer kurzlebigen eigenen — zu entscheiden je nachdem wie `nc` im
Aufrufkontext verfügbar ist):
- In `_subscribe_deploy_collect()`: einmal **vor** dem eigentlichen Deploy,
  Ergebnis in die Rückgabe/den späteren `experiment_metadata.json`-Eintrag
  aufnehmen (neuer Top-Level-Key `"clock_offset"`, analog zu `"power"`).
- In `run_baseline_block()` (Zeile 816): einmal pro Baseline-Block, Ergebnis
  in `baseline_summary.json` unter `"clock_offset"` ablegen (gleiche Struktur
  wie bei echten Experimenten, damit die Analyse beides einheitlich lesen kann).

### 3. Daten-Schema-Ergänzungen

`experiment_metadata.json` — neuer Key auf oberster Ebene:
```json
"clock_offset": {
  "offset_s": 0.0123,
  "round_trip_s": 0.0087,
  "offset_median_s": 0.0119,
  "offset_std_s": 0.0006,
  "n_pings": 7,
  "n_valid": 7,
  "measured_at": "2026-07-2xT..."
}
```
`baseline_summary.json` (aus `run_baseline_block()`) — derselbe Block, gleicher Key.

### 4. Analyse-Seite: `server/benchmark/analysis/analyze_benchmark.py`

Neue Funktion (Muster: wie `_local_baseline_power_w`/`_load_baseline_blocks`,
bereits im selben Stil vorhanden — dieselbe Trennung "Pipeline sammelt Rohdaten
+ Kalibrierwert, Analyse wendet die Korrektur an" beibehalten):

```python
def _offset_corrected_power_metrics(exp_dir: Path, meta: Dict, n_images: Optional[int]) -> Dict:
    """Re-integrate the raw power_*.csv over the inference window, shifted by
    the measured clock offset (T_pc_equiv = T_pi - offset_s), instead of
    trusting the Pi and PC clocks to already agree. Falls back to the
    uncorrected inference_window from metadata if no clock_offset was
    measured for this run (e.g. data collected before this feature existed).
    """
    offset = (meta.get("clock_offset") or {}).get("offset_s")
    if offset is None:
        return {}  # caller falls back to meta["power"]["inference_window"]

    res_csv = next(iter(exp_dir.glob("resource_usage_*.csv")), None)
    power_csv = next(iter(exp_dir.glob("power_*.csv")), None)
    if res_csv is None or power_csv is None:
        return {}

    # Pi-clock window -> PC-equivalent window (see sign convention in PLAN doc)
    import csv as _csv
    pi_times = []
    with open(res_csv, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                pi_times.append(float(row["timestamp_unix"]))
            except (ValueError, KeyError):
                continue
    if not pi_times:
        return {}
    t0_pc = min(pi_times) - offset
    t1_pc = max(pi_times) - offset

    ts, pw = _read_power_csv(power_csv)
    sel = [(t, p) for t, p in zip(ts, pw) if t0_pc <= t <= t1_pc]
    if len(sel) < 2:
        return {}
    energy = sum((pa + pb) / 2.0 * (tb - ta)
                 for (ta, pa), (tb, pb) in zip(sel, sel[1:]))
    duration = sel[-1][0] - sel[0][0]
    result = {
        "energy_J_offset_corrected": round(energy, 4),
        "avg_power_W_offset_corrected": round(energy / duration, 4) if duration > 0 else None,
        "duration_s_offset_corrected": round(duration, 3),
        "offset_applied_s": offset,
    }
    if n_images:
        result["energy_per_image_J_offset_corrected"] = round(energy / n_images, 6)
    return result
```

In `load_experiment()` nach dem bestehenden „Local (warm, time-bracketed)
baseline correction + EDP"-Block einhängen: Ergebnis-Dict in `row` mergen
(`row.update(_offset_corrected_power_metrics(exp_dir, meta, n_images))`),
plus `row["clock_offset_s"] = (meta.get("clock_offset") or {}).get("offset_s")`
und `row["clock_offset_round_trip_s"] = ...` als eigene, immer vorhandene
Diagnose-Spalten (auch wenn `None`) — damit man den gemessenen Versatz über
die ganze Kampagne plotten kann (Drift-Check, siehe unten).

**Offene Design-Entscheidung für die neue Session:** Soll die
**Netto-Energie mit lokaler Baseline** (bestehende `net_avg_power_W_local`/
`EDP`) zusätzlich **auch** offset-korrigiert werden (volle Korrektur:
Baseline lokal + Fenster offset-korrigiert), oder bleiben das zwei getrennt
ausgewiesene Korrekturen? Empfehlung: **kombinieren** — `EDP` sollte am Ende
auf Basis des offset-korrigierten Fensters *und* der lokalen Baseline
berechnet werden, da beide Korrekturen unabhängige, sich nicht
ausschließende Fehlerquellen adressieren. Das bedeutet, `_offset_corrected_power_metrics`
sollte im finalen Ausbau auch den `baseline_power_w`-Parameter (lokal)
entgegennehmen und direkt die Netto-Größe zurückgeben — beim ersten Entwurf
oben bewusst weggelassen, um die Korrekturen erst einzeln zu verifizieren.

### 5. Dokumentation

`MESSMETHODIK_ZEITSYNCHRONISATION.md` nach Fertigstellung erneut aktualisieren:
Abschnitt 3 („Fehlermodell") und 6 („Grenzen und Ausblick") von „nur
Bandbreite bekannt" auf „gemessen, siehe `clock_offset` pro Experiment"
umschreiben; Abschnitt 5 um einen neuen Audit-Punkt ergänzen: Vergleich der
gemessenen `offset_s`-Werte gegen die alte kausale Schranke (δ ∈
[−0,19s, +3,73s] aus der historischen Kampagne) als Konsistenz-Check.

---

## Umsetzungs-Reihenfolge (Checkliste für die neue Session)

1. [x] `client/deployment-receiver.py`: `handle_time_sync` + Subscription ergänzt.
2. [x] `benchmark_pipeline.py`: `_measure_clock_offset()` implementiert.
3. [x] `_subscribe_deploy_collect()` und `run_baseline_block()`: Aufruf eingehängt,
       Ergebnis in `experiment_metadata.json`/`baseline_summary.json` geschrieben.
4. [ ] **Echter Hardware-Test** (Pi + PC über NATS): mehrere Pings fahren, prüfen
       dass `offset_s` eine kleine, plausible Zahl ist (LAN: typischerweise
       einstellige bis niedrige zweistellige Millisekunden) und `round_trip_s`
       stabil bleibt. Kann nicht sinnvoll gemockt werden — echte Hardware nötig.
       **Noch offen** — nicht aus dieser Session heraus durchführbar.
5. [x] `analyze_benchmark.py`: `_offset_corrected_power_metrics()` implementiert,
       in `load_experiment()` eingehängt.
6. [x] Test mit synthetischen `power_*.csv`/`resource_usage_*.csv`-Fixtures und
       einem künstlich injizierten Offset (Skript im Session-Scratchpad,
       `test_offset_correction.py`): bestätigt, dass ein bekannter Zeitversatz
       (+3 s / −2,5 s / 0 s) korrekt herausgerechnet wird und ohne Korrektur
       eine deutliche Abweichung entsteht.
7. [x] Design-Entscheidung (Abschnitt 4 oben) geklärt: **kombiniert**, aber als
       zusätzliche Spalten (`net_avg_power_W_offset_corrected`,
       `EDP_offset_corrected`) statt Umbenennung/Ersetzung von `EDP` /
       `net_avg_power_W_local` — damit bleiben bestehende Notebooks/Figures,
       die auf `EDP` referenzieren, unverändert lauffähig; die vollständig
       korrigierte Größe steht parallel zur Verfügung.
8. [x] `MESSMETHODIK_ZEITSYNCHRONISATION.md` aktualisiert (Abschnitt 3, 5, 6
       und Referenztabelle).
9. [x] Syntax-Checks aller geänderten Dateien (`ast.parse`) — alle drei
       Dateien (`deployment-receiver.py`, `benchmark_pipeline.py`,
       `analyze_benchmark.py`) parsen fehlerfrei.

## Referenzen (Code-Stand zum Zeitpunkt dieses Plans)

| Was | Datei | Stelle |
|---|---|---|
| Pi NATS-Verbindung/Subscriptions | `client/deployment-receiver.py` | `start_consumer`, ab Z. 304 |
| PC Deploy+Collect | `server/benchmark/benchmark_pipeline.py` | `_subscribe_deploy_collect`, ab Z. 888 |
| Baseline-Block-Orchestrierung | `server/benchmark/benchmark_pipeline.py` | `run_baseline_block`, ab Z. 816 |
| Lokale Baseline-Korrektur (Vorbild für den Analyse-Teil) | `server/benchmark/analysis/analyze_benchmark.py` | `_local_baseline_power_w`, Z. 353 |
| Rohes Ressourcen-Fenster lesen (Vorbild) | `server/benchmark/benchmark_pipeline.py` | `_inference_window`, Z. 1127 |
| Trapezintegration (Vorbild) | `server/benchmark/benchmark_pipeline.py` | `_integrate`, Z. 1143 |
