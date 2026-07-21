# Testplan: Hardware-Validierung vor dem produktiven Sweep

**Erstellt:** 2026-07-21, vor dem ersten Durchlauf auf diesem Laptop (Erstinstallation,
noch kein Python, kein laufender Docker-Daemon, keine Datensätze extrahiert).
**Zweck:** Alle Code-Änderungen seit dem letzten Merge nach `main` (`f988a68`, "Merge
branch 'aml-submission'") mit **echtem Deployment auf Pi "Hans"** validieren, bevor der
produktive Sweep gestartet wird, der alle bisherigen `results/**`-Daten verwirft. Diese
Datei ist so geschrieben, dass sie einer neuen Session nach dem Testlauf übergeben werden
kann — sie enthält sowohl den Plan als auch die Stellen im Code, an denen die Ergebnisse
geprüft werden müssen.

**Nach dem Testlauf:** Trage Ergebnisse/Beobachtungen unten in Abschnitt 7 ein (oder lege
eine `TESTLAUF_ERGEBNIS_<datum>.md` daneben an) und gib diese Datei der neuen Session mit.

---

## 1. Was ist neu seit dem letzten Hardware-Benchmark (`f988a68` → `HEAD`)

Commits seit dem Merge: `7a3cb08` (roverlab dataset) … `b80db86` (HEAD, Lizenz). Die
funktional relevanten:

| Commit | Inhalt |
|---|---|
| `22ad6a4`, `9c99c16` | Power-Coverage-Audit + **Result-Attribution-Fix** (Experiment-ID-Tagging) |
| `1177309` | Confusion Matrix in der Analyse |
| `62d1739` | Cleanup, Rover-Lab-Modelle hinzugefügt |
| `d8b997c` | Bugfix: Experiment-ID eingeführt |
| `2bc7300` | **Rover-Lab-Szenarien fertiggestellt + Clock-Sync-Handshake implementiert** |
| `ff60cc7` | Baseline-Analyse: Unsicherheitsschätzung (SEM/CI) |
| `6974bc0`, `b80db86` | Housekeeping, Lizenz (nicht funktional) |

### 1.1 Result-Attribution-Fix (2026-07-20)

**Problem vorher:** Der Server akzeptierte die erste ZIP mit `merged_analysis*` als
Ergebnis des aktuell erwarteten Experiments — ohne zu prüfen, ob sie wirklich dazu
gehörte. Historisch führte das zu 3 von 1440 Läufen mit falsch zugeordneten (aber
Byte-identischen, bereits abgeschlossenen) Ergebnissen.

**Fix, an drei Stellen gemeinsam:**
- [`client/transmitter.py:55-74`](../../client/transmitter.py) `_read_experiment_tag()` —
  liest `experiment_id`/`scenario_name` aus dem lokal geschriebenen
  `experiment_config.json` und taggt jede NATS-Publikation mit
  `X-Experiment-Id`/`X-Scenario-Name`-Headern.
- [`server/benchmark/benchmark_pipeline.py:998-1066`](benchmark_pipeline.py) `handle_msg()`
  — routet jede eingehende ZIP anhand ihres Tags; nur bei Tag-Übereinstimmung gilt sie als
  Ergebnis des aktuell erwarteten Laufs. Eine falsch getaggte ZIP wird nie verworfen,
  sondern in den Ordner ihrer eigenen (bereits abgeschlossenen) Experiment-ID einsortiert.
- [`client/deployment-receiver.py:110-144`](../../client/deployment-receiver.py)
  `transmit_current_once()` — sendet die finalen Logs eines Laufs garantiert genau einmal
  (Lock + Flag), damit der "Prozess fertig"-Watcher und das "nächstes Deployment kam an"-
  Handling sich nicht gegenseitig doppelt auslösen.

**Zu prüfen im Testlauf:** Läuft der Result-Attribution-Fix wie erwartet, wenn mehrere
Experimente schnell hintereinander deployed werden? Siehe Abschnitt 7.2.

### 1.2 Clock-Offset-Handshake — **NIE auf echter Hardware getestet**

Siehe [`PLAN_ZEIT_HANDSHAKE.md`](PLAN_ZEIT_HANDSHAKE.md), Checkliste Punkt 4: als
"Implementiert, bis auf den echten Hardware-Test" markiert — der Autor der vorherigen
Session konnte das explizit **nicht** selbst prüfen, weil dafür PC+Pi live über NATS
laufen müssen. **Das ist der wichtigste Prüfpunkt dieses Testlaufs.**

- Pi-Seite: [`client/deployment-receiver.py:338-345`](../../client/deployment-receiver.py)
  `handle_time_sync()` — antwortet auf `time.sync.<device_id>` sofort mit der Pi-Uhrzeit.
- PC-Seite: [`benchmark_pipeline.py:900-935`](benchmark_pipeline.py) `_measure_clock_offset()`
  — NTP-artiger Round-Trip, 7 Pings, kleinster Round-Trip gewinnt.
- Wird vor jedem Experiment und jedem Baseline-Block aufgerufen
  ([`_subscribe_deploy_collect`](benchmark_pipeline.py) Zeile 963,
  `run_baseline_block` über `collect_result`) und in
  `experiment_metadata.json`/`baseline_summary.json` unter `"clock_offset"` gespeichert.
- Analyse-Seite: [`analyze_benchmark.py`](analysis/analyze_benchmark.py) `_offset_corrected_power_metrics()` —
  verschiebt das Inferenzfenster um den gemessenen Offset und berichtet
  `*_offset_corrected`-Spalten zusätzlich zu den unkorrigierten.

**Erwartung für ein LAN:** `offset_s` sollte eine kleine, plausible Zahl sein
(einstellige bis niedrige zweistellige Millisekunden), `round_trip_s` stabil über alle
7 Pings. Siehe Prüfschritte in Abschnitt 7.1.

### 1.3 Warme, lokale Baseline statt globaler kalter Baseline

`run_baseline_block()` ([`benchmark_pipeline.py:816-893`](benchmark_pipeline.py)) deployt
`n=3` Leerlauf-Durchläufe (`application.baseline_mode=true`, überspringt nur den
Klassifikator-Aufruf, ansonsten identischer Pfad) vor dem ersten und nach jeder
`(pca, samples_per_class)`-Zelle. Die Analyse ordnet jeder Zelle die zeitlich
nächstliegenden Baseline-Blöcke zu (`_bracketing_blocks`,
[`analyze_benchmark.py`](analysis/analyze_benchmark.py)) statt einer einzigen
Kalt-Start-Messung. Das bedeutet: **jede Zelle im Testlauf erzeugt automatisch
zusätzliche Deployments** (3 Baseline-Runs je Zellwechsel) — das zählt separat von den
"≥5 Zellen mit echtem Deployment", die unten für die eigentlichen Experimente gefordert
sind.

### 1.4 Chunked Deployment-Transfer für große Payloads

NATS `max_payload` ist auf 25 MB begrenzt (`middleware/nats.conf`). Für Imagenette mit
hohem `samples_per_class` (großer KNN-State) überschreitet die Deployment-ZIP das Limit.
Der Packager zerlegt große ZIPs jetzt automatisch in Chunks
(`X-Deployment-Chunked`-Header), `deployment-receiver.py:227-279` `message_handler()`
puffert Chunks pro `deployment_id` und setzt sie erst nach Empfang aller Teile zusammen.
**Wurde vermutlich schon in der alten Kampagne genutzt, aber nie mit dem aktuellen
Ergebnis-Tagging kombiniert getestet** — die Imagenette-Zelle im Testplan unten deckt das ab.

### 1.5 Rover-Lab-Szenarien — **komplett neu, noch nie deployed**

Drei neue Szenarien: `scenario_rover-lab_knn`, `_nearest_centroid`, `_kmeans` (binäre
Klassifikation `negative`/`obstacle`, Auflösung 160×160, aus echten Rover-Lab-Aufnahmen).
`server/benchmark/results/` enthält **keinen** `scenario_rover-lab_*`-Ordner — im
Unterschied zu allen anderen Szenarien wurden diese noch nie über NATS an einen Pi
deployt. Das Test-Setup unten (Abschnitt 4.5) deckt die Dataset-Extraktion und mindestens
einen echten Deploy-Zyklus für zwei der drei rover-lab-Szenarien ab.

### 1.6 Analyse-Erweiterungen

- **Confusion Matrix** (`analyze_benchmark.py` `_confusion_counts`/`build_confusion_table`)
  — liest `ground_truth`/`prediction` aus `merged_analysis_*.csv`, schreibt
  `results/analysis/confusion_matrix_per_class.csv`.
- **Baseline-Unsicherheitsschätzung** (`_local_baseline_sem_w`, `EDP_ci95_systematic`,
  `compute_suitability`) — neueste Änderung vor `HEAD`, noch nicht gegen echte neue Daten
  gegengeprüft.
- Power-Sampling-Rate: 5 Hz → 8 Hz (`dsp3005_reader.py`, `DEFAULT_RATE`), V/I-Setpoints
  werden nur einmal pro Session statt pro Zeile gelesen.

---

## 2. Bekannte Schwachstellen / Risiken (während des Testlaufs beobachten)

Diese sind beim Code-Review aufgefallen und sollten während bzw. nach dem Testlauf aktiv
geprüft werden — nicht nur "funktioniert es", sondern "funktioniert es aus dem
genannten Grund plausibel richtig":

1. **Hardcodierte NATS-IP passt nicht zu diesem Laptop.**
   [`client/deployment-receiver.py:35`](../../client/deployment-receiver.py) und
   [`client/transmitter.py:13`](../../client/transmitter.py) haben
   `NATS_URL = "nats://192.168.178.62:4222"` fest einprogrammiert. Die aktuelle IPv4 dieses
   Laptops ist `192.168.179.4` (**anderes Subnetz** — 178 vs. 179). Muss vor dem Kopieren
   auf den Pi angepasst werden, siehe Abschnitt 4.2.
2. **`export_rover-lab.py` lädt standardmäßig von Hugging Face, nicht aus dem bereits im
   Repo liegenden `rover-lab.tar.gz`.** Ohne Workaround braucht der erste Lauf
   Internetzugang zu `huggingface.co/datasets/Fundulus/rover-lab`. Workaround in
   Abschnitt 4.5.
3. **`EXCLUDED_EXPERIMENTS` in `analyze_benchmark.py` ist eine hartcodierte Liste von 3
   IDs** aus der alten (verworfenen) Kampagne. Sie greift lautlos ins Leere, sobald die
   Ergebnisstruktur sich ändert (kein Fehler, einfach 0 Ausschlüsse) — für den
   *produktiven* Sweep nach diesem Test sollte diese Liste geleert/neu bewertet werden,
   da der Result-Attribution-Fix genau den Fehlermodus strukturell verhindert, der zu
   diesen 3 Ausschlüssen führte.
4. **Zwei Definitionen von `ensure_feature_processor()`** in `benchmark_pipeline.py`
   (Zeile 454 und Zeile 507 mit unterschiedlicher Signatur) — Python verwendet nur die
   zweite, die erste ist toter Code. Funktional unkritisch (nicht erreichbar), aber
   verwirrend bei künftigen Änderungen.
5. **`_legacy_run_pipeline()` (Zeile 1289ff.) ruft Funktionen mit veralteten Signaturen
   auf** (z. B. `create_initial_state(samples_per_class, …)` ohne `ctx`) und würde bei
   Aufruf sofort crashen — ist aber tatsächlich **nirgendwo mehr referenziert** (toter
   Code, nur über `main()` → `run_pipeline()` → `_run_scenario()` läuft der aktive Pfad).
   Nicht testrelevant, aber Aufräum-Kandidat vor der Veröffentlichung.
6. **`_local_baseline_sem_w`** kombiniert die Standardfehler von zwei benachbarten
   Baseline-Blöcken über `sqrt(sem_before²+sem_after²)/2` — eine nicht ganz konventionelle
   Formel für "Mittelwert zweier unabhängiger Schätzungen" (im Code selbst als bewusste
   Wahl kommentiert). Rechnerisch in sich konsistent, aber vor Verwendung im Paper von
   einer zweiten Person gegenprüfen lassen.
7. **Kein Python, kein laufender Docker-Daemon, keine Datensätze auf diesem Laptop** —
   siehe Abschnitt 4, komplettes Setup nötig, nicht nur Konfiguration.
8. **`baseline_power_w`/`--measure-baseline` sind Altlast eines abgelösten Mechanismus**
   (siehe `analyze_benchmark.py:589,591` — dort bereits selbst als `# global-baseline,
   legacy` markiert). Die neue, automatische Zellen-Baseline (`run_baseline_block`,
   `net_avg_power_W_local`/`EDP`) hat das ersetzt. Für diesen Testlauf bewusst
   `baseline_power_w: null` gelassen (Abschnitt 6.1). Separates Aufräumen in
   [`PROMPT_BASELINE_CLEANUP.md`](PROMPT_BASELINE_CLEANUP.md) vorgesehen — dabei fällt auch
   der zugehörige, seit dem Refactor nicht mehr aktualisierte Abschnitt in
   `analysis/README.md` (Zeilen 89-138, 288) mit weg, der die Legacy-Metrik noch als
   "headline energy metric" bezeichnet.

---

## 3. Testziel

**Mindestens 5 Experiment-Zellen mit echtem Deployment auf Hans**, so gewählt, dass jede
der obigen Änderungen mindestens einmal durchlaufen wird:

| # | Szenario | Zweck | PCA | spc | seed |
|---|---|---|---|---|---|
| 1 | `scenario_cifar10_knn_benchmark` | Schneller Sanity-Check, kleine Auflösung (32×32) | nopca | 2 | 42 |
| 2 | `scenario_cifar10_nearest_centroid` | Zweiter Klassifikator-Typ, PCA-Pfad + Feature-Processor-Caching | pca95var | 5 | 42 |
| 3 | `scenario_cifar10_kmeans` | Dritter Klassifikator-Typ | nopca | 2 | 42 |
| 4 | `scenario_imagenette_knn` | Große Auflösung (224×224) → testet **Chunked Transfer** | pca95var | 2 | 42 |
| 5 | `scenario_rover-lab_knn` | **Neues, nie deploytes Szenario**, 160×160, binäre Klassen | nopca | 2 | 42 |
| 6 | `scenario_rover-lab_nearest_centroid` | Rover-Lab + PCA-Feature-Processor-Training aus dem Stand | pca95var | 5 | 42 |

Jeder Zellwechsel löst automatisch einen Baseline-Block (3 zusätzliche Deployments) aus
— macht real ca. 6 × 4 = 24 Deployments auf Hans, nicht nur 6. Das ist gewollt: es prüft
den Clock-Handshake und das Ergebnis-Tagging deutlich öfter als nur 5 Mal.

`scenario_rover-lab_kmeans` ist im Testlauf bewusst ausgelassen (dritter rover-lab-Typ),
da 2 von 3 rover-lab-Klassifikatoren ausreichen, um sowohl den Dataset-Export als auch
den PCA-Pfad für das neue Szenario abzudecken; `_kmeans` folgt automatisch im
produktiven Sweep, sobald 1 und 2 sauber durchgelaufen sind.

---

## 4. Setup auf diesem Laptop (Ersteinrichtung — noch nichts vorhanden)

Geprüfter Ist-Zustand: **kein Python installiert** (nur der Windows-Store-Stub-Alias),
Docker-CLI vorhanden aber **Docker-Desktop-Daemon läuft nicht**, alle `dataset_*/train`
und `dataset_*/test`-Ordner fehlen (per `.gitignore` nie versioniert), aktuelle
Laptop-IP `192.168.179.4`. Netzteil ist laut Vorgabe bereits per USB mit diesem PC
verbunden; Pi "Hans" ist im gleichen Netz mit `hans`/`samplesizeknn` erreichbar.

### 4.1 Python installieren

Python **3.11.x** (passend zu Pi OS und `.devcontainer/requirements.txt`) von
python.org installieren (64-bit, "Add to PATH" aktivieren). Danach neues Terminal öffnen
und prüfen:

```powershell
python --version
```

### 4.2 Projekt-venv anlegen und Pakete installieren

```powershell
cd "C:\Users\Philipp\Documents\00_TinyMLOps-Benchmark\TinyTrail\server\benchmark"
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install --upgrade pip
pip install `
  tensorflow==2.15.0 `
  numpy==1.24.2 `
  scikit-learn==1.4.2 `
  opencv-python-headless==4.9.0.80 `
  Pillow==10.3.0 `
  psutil==5.9.8 `
  scipy==1.12.0 `
  pyserial `
  "nats-py>=2.11.0,<3.0.0" `
  "aiofiles>=24.1.0,<25.0.0"
```

(`pyserial` und `nats-py`/`aiofiles` stehen nicht in `.devcontainer/requirements.txt` —
die sind für den Trainings-Container gedacht, nicht für die Pipeline selbst. Sie werden
hier zusätzlich gebraucht, weil `benchmark_pipeline.py` den Deployment-Packager per
`sys.executable` **im selben venv** aufruft, nicht über dessen eigenes Poetry-Env, und
weil `dsp3005_reader.py` `pyserial` direkt importiert.)

Danach in dieser Datei den Pfad zu `$py` für alle folgenden Befehle setzen:

```powershell
$py = "C:\Users\Philipp\Documents\00_TinyMLOps-Benchmark\TinyTrail\server\benchmark\.venv311\Scripts\python.exe"
```

### 4.3 NATS JetStream Middleware (Docker)

Docker Desktop starten (Daemon lief beim Check noch nicht). Dann:

```powershell
docker run -d --name nats-jetstream `
  -p 4222:4222 -p 8222:8222 `
  -v "C:\Users\Philipp\Documents\00_TinyMLOps-Benchmark\TinyTrail\middleware\nats.conf:/etc/nats/nats.conf" `
  -v tinytrail_nats_data:/data `
  nats:latest -c /etc/nats/nats.conf
```

Hinweis: Der Befehl in `middleware/readme.md` ist unvollständig (der `-v`-Mount hat dort
keinen Container-Zielpfad `:/etc/nats/nats.conf` und keinen Datenmount für den
`jetstream.store_dir: /data` aus `nats.conf`) — die obige Version behebt beides und
sorgt zusätzlich dafür, dass JetStream-Daten einen Neustart des Containers überleben.

Prüfen:

```powershell
docker ps
curl http://localhost:8222/varz
```

### 4.4 IP-Adresse in den Client-Dateien korrigieren

`ipconfig` erneut ausführen und die aktuelle IPv4 bestätigen (DHCP kann sie geändert
haben). Dann in **beiden** Dateien `NATS_URL` auf diese IP setzen:

- [`client/deployment-receiver.py:35`](../../client/deployment-receiver.py)
- [`client/transmitter.py:13`](../../client/transmitter.py)

```python
NATS_URL = "nats://<aktuelle-Laptop-IP>:4222"
```

Prüfen, dass `client/config/device_id.txt` weiterhin `my_device_01` enthält (passend zu
`"device_id": "my_device_01"` in `benchmark_config.json`).

### 4.5 Datensätze exportieren (alle drei fehlen komplett auf diesem Laptop)

```powershell
& $py "server\Retrainer\on-device-drift-detection-main\examples\dataset_cifar10\export_cifar10.py"
& $py "server\Retrainer\on-device-drift-detection-main\examples\dataset_imagenette\export_imagenette.py" --size 320
```

Für Rover-Lab: das Archiv liegt bereits lokal im Repo
(`dataset_rover-lab/rover-lab.tar.gz`, 16,7 MB) — `export_rover-lab.py` würde es aber
trotzdem erneut von Hugging Face herunterladen, weil es nur unter
`dataset_rover-lab/downloads/rover-lab.tar.gz` nach einer Kopie sucht. Um den
Internet-Download zu vermeiden (Risiko #2 oben):

```powershell
cd "server\Retrainer\on-device-drift-detection-main\examples\dataset_rover-lab"
New-Item -ItemType Directory -Force downloads | Out-Null
Copy-Item rover-lab.tar.gz downloads\rover-lab.tar.gz
& $py export_rover-lab.py
cd "C:\Users\Philipp\Documents\00_TinyMLOps-Benchmark\TinyTrail"
```

Falls das nicht funktioniert (z. B. weil das Archiv-Format nicht exakt passt), ersatzweise
mit Internetzugang laufen lassen — es lädt dann von `huggingface.co/datasets/Fundulus/rover-lab`.

### 4.6 Pi "Hans" vorbereiten

```powershell
ssh hans@hans.local
```

(Falls mDNS/`hans.local` nicht auflöst: IP über den Router oder `arp -a` im gleichen
Subnetz ermitteln und direkt verwenden.)

Aktualisierte Client-Dateien auf den Pi kopieren (**wichtig**: `deployment-receiver.py`
und `transmitter.py` wurden seit dem letzten Lauf auf Hans geändert — Result-Tagging +
Zeit-Handshake + Chunk-Reassembly sind neu):

```powershell
scp client\deployment-receiver.py hans@hans.local:/home/hans/client/deployment-receiver.py
scp client\transmitter.py hans@hans.local:/home/hans/client/transmitter.py
```

Auf dem Pi prüfen, dass die neue Version ankam:

```bash
grep -n "handle_time_sync\|X-Deployment-Chunked" /home/hans/client/deployment-receiver.py
```

Für den Testlauf reicht ein sichtbarer `tmux`-Lauf (volle Logs sehen, statt gleich als
Service zu verstecken):

```bash
tmux new -s receiver
cd /home/hans/client
python3 deployment-receiver.py
```
(Detach: `Ctrl+B`, dann `D`.)

### 4.7 Voltcraft DSP-3005 (Netzteil) — COM-Port ermitteln

Netzteil ist laut Vorgabe bereits angeschlossen. COM-Port im Windows-Geräte-Manager unter
"Anschlüsse (COM & LPT)" ablesen (nicht zwingend `COM3`, das ist nur der Default). Dann
in einer **separaten Test-Config** (Abschnitt 5) `power.port` entsprechend setzen. Falls
die "Power Supply PC Monitor"-App offen ist, blockiert sie den Port — vorher schließen.

Direkter Verbindungstest, bevor der große Testlauf startet:

```powershell
& $py dsp3005_reader.py COM<N>
```
(Ctrl+C zum Beenden; sollte V/I/P live anzeigen.)

---

## 5. Test-Konfiguration (separat von der produktiven `benchmark_config.json`)

Neue Datei `server/benchmark/test_config_validation.json` anlegen (nicht die
produktive `benchmark_config.json` überschreiben):

```json
{
  "device_id": "my_device_01",
  "nats_url": "nats://localhost:4222",

  "experiment_names": [
    { "name": "scenario_cifar10_knn_benchmark",
      "pca_configs": [{ "enabled": false }],
      "samples_per_class": [2], "seeds": [42] },
    { "name": "scenario_cifar10_nearest_centroid",
      "pca_configs": [{ "enabled": true, "explained_variance": 0.95 }],
      "samples_per_class": [5], "seeds": [42] },
    { "name": "scenario_cifar10_kmeans",
      "pca_configs": [{ "enabled": false }],
      "samples_per_class": [2], "seeds": [42] },
    { "name": "scenario_imagenette_knn",
      "pca_configs": [{ "enabled": true, "explained_variance": 0.95 }],
      "samples_per_class": [2], "seeds": [42] },
    { "name": "scenario_rover-lab_knn",
      "pca_configs": [{ "enabled": false }],
      "samples_per_class": [2], "seeds": [42] },
    { "name": "scenario_rover-lab_nearest_centroid",
      "pca_configs": [{ "enabled": true, "explained_variance": 0.95 }],
      "samples_per_class": [5], "seeds": [42] }
  ],

  "result_timeout_seconds": 1800,
  "abort_on_timeout": true,
  "skip_completed": true,

  "test_images_per_class": 5,

  "test_data_dirs": {
    "rover-lab": "server/Retrainer/on-device-drift-detection-main/examples/dataset_rover-lab/test"
  },

  "power": {
    "enabled": true,
    "required": true,
    "port": "COM<N-aus-4.7>",
    "rate": 8.0
  }
}
```

`test_images_per_class: 5` statt der produktiven 20, damit der Testlauf schnell bleibt
(v. a. Imagenette 224×224 ist auf dem Pi Zero 2W langsam).

---

## 6. Durchführung

### 6.1 Idle-Baseline messen — **entfällt, Feld wurde entfernt**

Die alte, manuelle, globale Baseline-Korrektur (`--measure-baseline`-Flag,
`power.baseline_power_w`-Config-Feld, `net_avg_power_W`/`net_energy_per_image_J`) ist seit
[`PROMPT_BASELINE_CLEANUP.md`](PROMPT_BASELINE_CLEANUP.md) vollständig aus dem Code entfernt
(`benchmark_pipeline.py`, `analysis/analyze_benchmark.py`, `benchmark_config.json`). Es gibt
also weder das CLI-Flag noch das Config-Feld mehr — kein manueller Messschritt nötig. Die
**einzige** Baseline-Korrektur, auf die es ankommt (`net_avg_power_W_local`/`EDP`), kommt
automatisch aus den `run_baseline_block()`-Leerlaufläufen, die ohnehin vor/nach jeder Zelle
passieren.

### 6.2 Dry-Run (prüft Pfade/Konfiguration, ohne den Pi anzufassen)

```powershell
& $py benchmark_pipeline.py --config test_config_validation.json --results-dir results_validation --dry-run
```

Erwartung: alle 6 Szenarien lösen ohne `FileNotFoundError` auf (Modell, Config,
Test-Split). Falls `scenario_rover-lab_*` hier scheitert, liegt es fast immer am
Dataset-Export aus Schritt 4.5 (train/test fehlt) oder an `test_data_dirs`.

### 6.3 Echter Testlauf

```powershell
& $py benchmark_pipeline.py --config test_config_validation.json --results-dir results_validation
```

**Wichtig:** `--results-dir results_validation` (nicht der produktive `results/`-Ordner)
— dieser Testlauf ist explizit dazu da, verworfen zu werden, bevor der produktive Sweep
mit `results/` beginnt (siehe Abschnitt 8).

Während des Laufs beobachten:
- Pi-Terminal (`tmux attach -t receiver`): keine Python-Tracebacks, `Received deployment
  chunk …` bei Imagenette sichtbar (Chunking greift).
- PC-Konsole: `[RESULT] CSV received: …` pro Zelle, **keine** `[RESULT] WARNING: ZIP
  carries no X-Experiment-Id header`-Zeile (falls die doch auftaucht: Pi läuft noch mit
  altem `transmitter.py` — Schritt 4.6 wiederholen).
- `[BASELINE:…]` Zeilen vor/nach jedem Zellwechsel.

### 6.4 Analyse auf dem isolierten Testordner laufen lassen

```powershell
& $py analysis\analyze_benchmark.py --results-dir results_validation --out-dir results_validation\analysis
& $py analysis\verify_power_coverage.py results_validation --csv-out results_validation\analysis\power_coverage_audit.csv
```

(Beide Skripte akzeptieren ein eigenes Ergebnisverzeichnis — die produktiven
`results/`-Daten bleiben dabei unberührt.)

---

## 7. Nach dem Testlauf zu prüfende Punkte (Checkliste)

Trage hier direkt die Ergebnisse ein oder verlinke eine separate Ergebnisdatei — die
neue Session braucht das, um die Analyse fortzusetzen.

### 7.1 Clock-Offset-Handshake (höchste Priorität, siehe 1.2)

- [ ] Öffne 2-3 `experiment_metadata.json` aus `results_validation/<scenario>/exp_*/`.
      Ist `clock_offset.offset_s` vorhanden und plausibel (LAN: ms-Bereich, nicht
      Sekunden)?
- [ ] Ist `clock_offset.round_trip_s` über die 6 Zellen stabil (keine Ausreißer > 100 ms)?
- [ ] Öffne ein `baseline_XXX/baseline_summary.json` — ist dort ebenfalls ein plausibler
      `clock_offset`-Block vorhanden?
- [ ] Ergibt `analyze_benchmark.py`s `energy_J_offset_corrected` einen Wert, der nahe an
      `power.inference_window.energy_J` (unkorrigiert) liegt? Große Abweichung würde auf
      ein Problem im Vorzeichen/der Fensterlogik hindeuten (siehe
      `_offset_corrected_power_metrics`, Abschnitt 1.2).

### 7.2 Result-Attribution / Ergebnis-Tagging

- [ ] Keine `[RESULT] WARNING: ZIP carries no X-Experiment-Id header`-Meldung in der
      PC-Konsolenausgabe des Laufs.
- [ ] Jeder `exp_*`-Ordner unter `results_validation/<scenario>/` enthält genau eine
      `merged_analysis_*.csv`, die zum jeweiligen `experiment_id` passt (Dateiname/`run_id`
      innerhalb der CSV vergleichen).
- [ ] Keine "Attributed stray result to experiment …"-Meldung — falls doch: prüfen, ob
      der zugehörige Zielordner (`results_validation/<anderes-scenario>/<id>/`) korrekt
      angelegt wurde statt Daten zu verlieren.

### 7.3 Rover-Lab (neu, nie deployt)

- [ ] `scenario_rover-lab_knn` und `scenario_rover-lab_nearest_centroid` haben je einen
      vollständigen `exp_*`-Ordner mit `experiment_metadata.json` + `merged_analysis*.csv`.
- [ ] `classifier_type` in `experiment_metadata.json` stimmt mit dem erwarteten Typ überein
      (`LightweightKNN` bzw. `NearestCentroidClassifier`).
- [ ] `inference_resolution` = `[160, 160]` wie in `config.json` erwartet.
- [ ] PCA-Feature-Processor wurde für `scenario_rover-lab_nearest_centroid` korrekt aus
      `feature_processor.npz` abgeleitet/gecacht (`feature_processor_pca95var.npz` im
      `model/`-Ordner vorhanden, `pca_explained_variance_%` in der Metadatei plausibel
      nahe 95).
- [ ] `merged_analysis_*.csv` enthält nur die zwei Klassen `negative`/`obstacle` in
      `ground_truth`/`prediction`.

### 7.4 Chunked Transfer (Imagenette)

- [ ] Pi-Log zeigt `Received deployment chunk N/M` mehrfach für die Imagenette-Zelle.
- [ ] Trotzdem vollständiges, unbeschädigtes Deployment (Pi konnte `main.py` importieren
      und ausführen, kein `ModuleNotFoundError`).

### 7.5 Formate / erwartete Dateien je `exp_*`-Ordner

Soll-Zustand laut `README.md` § "Result Layout":
```
merged_analysis_*.csv   resource_usage_*.csv   performance_*.json
drift_detection_*.csv   power_exp_*.csv        experiment_metadata.json
```
- [ ] Für jede der 6 Zellen: alle sechs Dateitypen vorhanden (fehlt `drift_detection_*.csv`,
      ist das ok — alle Testszenarien haben `"drift_detectors": []`, siehe `config.json`).
- [ ] `experiment_metadata.json` enthält alle in Abschnitt 1 genannten Schlüssel,
      insbesondere `clock_offset` und `power.inference_window`.
- [ ] `results_validation/benchmark_summary.json` und je
      `results_validation/<scenario>/benchmark_summary.json` wurden geschrieben und
      `"ok"` zählt alle 6 Zellen (kein `"timeout"`).

### 7.6 Analyse-Ausgaben

- [ ] `results_validation/analysis/experiment_summary.csv` enthält 6 Zeilen (eine je Zelle).
- [ ] `results_validation/analysis/confusion_matrix_per_class.csv` enthält Zeilen für
      CIFAR-10 (10 Klassen), Imagenette (10 Klassen) und Rover-Lab (2 Klassen).
- [ ] `power_coverage_audit.csv` (`verify_power_coverage.py`) zeigt für alle 6 Läufe
      Punkte 1-4 (Ränder, kausale Einbettung, Lückenlosigkeit, Zuordnung) als PASS.

---

## 8. Vor dem produktiven Sweep-Start

Sobald Abschnitt 7 abgehakt und plausibel ist:

1. `test_config_validation.json` und `results_validation/` können gelöscht werden — sie
   waren nur zur Validierung gedacht.
2. Produktive `benchmark_config.json` mit dem richtigen `power.port` aktualisieren
   (`baseline_power_w` gibt es nicht mehr, siehe Schritt 6.1 — nichts einzutragen).
3. Alte, **vor dem Result-Attribution-Fix erhobene** Daten in `server/benchmark/results/`
   verwerfen (laut `README.md` ohnehin geplant — "All experiments are being fully re-run
   regardless"):
   ```powershell
   & $py benchmark_pipeline.py --clean-results --dry-run   # erst zur Kontrolle
   & $py benchmark_pipeline.py --clean-results              # danach wirklich starten
   ```
4. `EXCLUDED_EXPERIMENTS` in `analysis/analyze_benchmark.py` leeren/neu bewerten (Risiko
   #3 in Abschnitt 2) — die 3 dort gelisteten IDs gehören zur alten, verworfenen Kampagne.
5. `deployment-receiver.py` auf Hans von `tmux` auf den in `README.md` beschriebenen
   `systemd`-Service umstellen, damit der produktive Sweep (voraussichtlich 720+
   Experimente über Stunden/Tage) nicht von einer geschlossenen SSH-Sitzung abhängt.

---

## 9. Offene Punkte für die neue Session

- Ergebnisse aus Abschnitt 7 hier (oder in einer verlinkten Datei) eintragen, bevor die
  neue Session startet.
- Falls der Clock-Handshake (7.1) unplausible Werte zeigt (z. B. Sekunden statt
  Millisekunden): das deutet auf eine blockierende Operation vor der Zeitstempel-Erfassung
  in `handle_time_sync()` hin (siehe Warnung in `PLAN_ZEIT_HANDSHAKE.md` Abschnitt 1) —
  zuerst dort nachsehen, nicht die Korrekturformel in `analyze_benchmark.py` in Frage
  stellen.
- Falls Rover-Lab-Läufe scheitern: zuerst `test_data_dirs`/Dataset-Export prüfen (Risiko
  #2), erst danach den Szenario-Code selbst verdächtigen — die Konfiguration ist neuer
  als der zugrunde liegende Harness-Code.
