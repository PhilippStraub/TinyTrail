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
| 1 | `scenario_cifar10_knn` | Schneller Sanity-Check, kleine Auflösung (32×32) | nopca | 2 | 42 |
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
    { "name": "scenario_cifar10_knn",
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

## 7. Nach dem Testlauf zu prüfende Punkte (Checkliste) — **ausgefüllt, Testlauf durchgeführt 2026-07-21**

Alle 6 Zellen liefen erfolgreich (`Pipeline complete: 6/6 experiments succeeded`), nach dem
Rover-Lab-Fix (10.8) im zweiten Anlauf. Ergebnisse unten sind aus dem tatsächlichen Lauf.

### 7.1 Clock-Offset-Handshake (höchste Priorität, siehe 1.2) — ✅ funktioniert, ⚠️ Wert überraschend groß

- [x] `clock_offset.offset_s` vorhanden in allen `experiment_metadata.json` UND
      `baseline_summary.json` (7 gültige Pings von 7 in jedem Fall geprüft).
- [x] `round_trip_s` stabil und klein (7-12 ms über alle 6 Zellen) — kein Netzwerkproblem.
- [ ] **Nicht wie erwartet**: `offset_s` liegt bei konstant **≈ -0,248 s** (nicht ms-Bereich!),
      über alle 6 Zellen praktisch identisch (-0,2477 bis -0,2482, Streuung < 1 ms). Das ist
      kein Messfehler (Round-Trips exzellent, Wert extrem reproduzierbar) — die Pi-Uhr liegt
      real ~248 ms hinter der PC-Uhr. Liegt knapp außerhalb der historischen kausalen
      Schranke [-0,19s, +3,73s] aus der alten Kampagne (Abschnitt 5.1 MESSMETHODIK) — nach
      dortigem Konsistenz-Check (Punkt 7) ein Hinweis, dem nachzugehen sich lohnt, auch wenn
      die alte Schranke von anderer Hardware/Kampagne stammt und nicht 1:1 vergleichbar ist.
- [x] Offset-Korrektur wirkt sich messbar aus: `net_avg_power_W_local` vs.
      `net_avg_power_W_offset_corrected` unterscheiden sich um 5-11 % bei den kurzen
      CIFAR-10/Rover-Lab-Läufen, aber nur ~0,4 % bei Imagenette (langes Fenster) — exakt das
      in der Doku vorhergesagte 1/T-Skalierungsverhalten. Handshake ist damit sowohl technisch
      funktionsfähig als auch inhaltlich sinnvoll — validiert den ganzen Sinn der Änderung.

### 7.2 Result-Attribution / Ergebnis-Tagging — ✅ vollständig bestätigt

- [x] Keine einzige `WARNING: ZIP carries no X-Experiment-Id header`-Meldung im gesamten Lauf
      (0 Treffer über beide Log-Dateien) — jede Übertragung war korrekt getaggt.
- [x] Genau ein reales Race-Ereignis beobachtet und korrekt behandelt:
      `baseline_001_run1`s Ergebnis kam verspätet an, während bereits auf
      `baseline_001_run2` gewartet wurde — wurde korrekt seinem eigenen (bereits
      abgeschlossenen) Ordner zugeordnet, nicht verworfen, nicht fälschlich
      `baseline_001_run2` zugeschrieben. Das ist exakt der Fehlermodus, den der Fix
      beheben sollte — hier live unter echten Timing-Bedingungen bestätigt.

### 7.3 Rover-Lab (neu, nie deployt) — ✅ nach Bugfix (10.8) erfolgreich

- [x] Beide Rover-Lab-Zellen (`scenario_rover-lab_knn`, `_nearest_centroid`) haben
      vollständige `exp_*`-Ordner mit `experiment_metadata.json` + `merged_analysis*.csv`.
- [x] `classifier_type` korrekt (`LightweightKNN` / `NearestCentroidClassifier`).
- [x] `inference_resolution` = `[160, 160]`.
- [x] PCA-Feature-Processor für `_nearest_centroid` korrekt erzeugt/gecacht.
- [x] `confusion_matrix_per_class.csv` enthält für beide Rover-Lab-Zellen genau 2 Zeilen
      (Klassen `negative`/`obstacle`, `support=5` je Klasse, passend zu
      `test_images_per_class=5`).
- ⚠️ **War vor dem Fix in 10.8 komplett blockiert** (Absturz vor jedem Deployment) — ohne
      diesen Testlauf wäre der Copy-Paste-Bug erst beim produktiven Sweep aufgefallen.

### 7.4 Chunked Transfer (Imagenette) — ⚠️ **nicht getestet, nicht in diesem Testlauf abgedeckt**

- [ ] 0 Treffer für "chunk"/"Received deployment chunk" im gesamten Log — die Chunking-Logik
      wurde in diesem Testlauf **nicht ausgelöst**. Grund: Die Testzelle nutzte bewusst ein
      kleines `samples_per_class=2` (für einen schnellen Validierungslauf), wodurch die
      Deployment-ZIP klein genug blieb, um unter dem NATS-`max_payload`-Limit (25 MB) zu
      bleiben. Chunking greift laut README erst bei hohem `samples_per_class` in Kombination
      mit Imagenette. **Das ist eine echte Lücke dieses Testlaufs** — die Chunk-Reassemblierung
      auf dem Pi (`deployment-receiver.py:227-279`) bleibt bis zum produktiven Sweep (oder
      einem gezielten Nachtest mit z. B. `samples_per_class=100`) ungetestet.

**Nachtrag (Folgesession, 2026-07-21, gezielter Nachtest gegen Hans):** Getestet mit
`samples_per_class=100` (dem höchsten Wert, den `benchmark_config.json` überhaupt
vorsieht) sowohl mit `pca95var` (test_images_per_class=5) als auch mit `nopca`/vollem
1280-dim-Feature-Vektor (`test_images_per_class=20`, produktiver Wert) — beide echte
Deployments auf Hans, beide fehlerfrei, aber:

| Variante | spc | test_images/class | ZIP-Größe |
|---|---|---|---|
| `pca95var` | 100 | 5 | 12,08 MB |
| `nopca` | 100 | 20 | 12,76 MB |

Beide bleiben weit unter den 25 MB — **selbst die ungünstigste innerhalb der produktiven
Config erreichbare Kombination (kein PCA, `spc=100`, volle Testbildzahl) löst kein
Chunking aus.** Die Annahme oben ("Chunking greift bei hohem `samples_per_class`") ist
für die tatsächlich konfigurierten Werte falsch — `samples_per_class` geht in der
produktiven Config nie über 100 hinaus (Abschnitt 5/8.2-Liste `[2,5,...,100]`). Die
Chunk-Reassemblierung bleibt damit **strukturell ungetestet** für den gesamten
produktiven Sweep, nicht nur für diesen einen Testlauf. Auf Nutzerentscheidung: kein
synthetischer Extremtest mit unrealistischem `spc` (z. B. 400+) durchgeführt — das
Risiko wird bewusst dokumentiert und in Kauf genommen, statt mit nicht
repräsentativen Parametern nachzutesten. Für eine spätere Session, falls das
noch geschlossen werden soll: `samples_per_class` deutlich über 100 (Imagenette-Klassen
haben min. 858 Trainingsbilder, spc bis ~800 wäre möglich) nur als isolierter,
nicht-produktiver Testlauf, nie in der produktiven Config.

### 7.5 Formate / erwartete Dateien je `exp_*`-Ordner — ✅ vollständig

- [x] Alle erwarteten Dateitypen für jede der 6 Zellen vorhanden (kein `drift_detection_*.csv`
      — korrekt, da `"drift_detectors": []` in allen Test-Configs).
- [x] `experiment_metadata.json` enthält alle erwarteten Schlüssel inkl. `clock_offset` und
      `power.inference_window`.
- [x] `benchmark_summary.json` (root + je Szenario) geschrieben, alle 6 Zellen `"ok"`, kein
      `"timeout"` (0 Treffer für "TIMEOUT" im gesamten Log).

### 7.6 Analyse-Ausgaben — ✅, mit einem neuen Befund

- [x] `experiment_summary.csv`: 6 Zeilen, eine je Zelle.
- [x] `confusion_matrix_per_class.csv`: 44 Zeilen (10+10+10+10+2+2 Klassen über die 6 Zellen).
- ⚠️ `power_coverage_audit.csv`: **0 PASS, 42 WARN, 0 FAIL** (42 = 6 Experimente + 36
      Baseline-Läufe). Für die 6 echten Experimente sind die WARNs die erwarteten (Punkte 5
      Kreuzkorrelation-Rauschen + Punkt 6 Fenster-Sensitivität, laut Doku kein Messfehler —
      Sensitivität hier ungewöhnlich hoch, 10-84 %, weil dieser Testlauf bewusst nur
      `test_images_per_class=5` nutzt und damit extrem kurze Inferenzfenster erzeugt, nicht
      repräsentativ für den produktiven Sweep mit 20/Klasse). **Neuer, echter Befund:** Für
      **alle 36 Baseline-Läufe** meldet der Audit
      `no started_at/result_received_at in metadata -> containment check skipped` — die
      kausale Einbettungsprüfung (Punkt 2, die "tragende Aussage" des ganzen Dokuments) läuft
      für Baseline-Blöcke gar nicht, weil `run_baseline_block()` diese beiden Zeitstempel
      nicht in `baseline_summary.json` schreibt (im Gegensatz zu `experiment_metadata.json`,
      wo sie vorhanden sind). Das bestätigt empirisch eine Lücke, die
      `MESSMETHODIK_ZEITSYNCHRONISATION.md` Abschnitt 6 bereits theoretisch benannt hatte
      ("Ließe sich mit geringem Aufwand in `verify_power_coverage.py` ergänzen"). Für die
      neue Session: `started_at`/`result_received_at` pro Baseline-Run in
      `run_baseline_block()` mitschreiben, damit der Audit auch Baseline-Läufe vollständig
      prüfen kann.

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
  **Nachtrag (Folgesession):** Diese Vermutung geprüft und verworfen — sowohl
  `handle_time_sync()` ([client/deployment-receiver.py:354-359](../../client/deployment-receiver.py))
  als auch `_measure_clock_offset()` ([benchmark_pipeline.py:915-943](benchmark_pipeline.py))
  erfassen ihre Zeitstempel bereits unmittelbar (keine Awaits/Blockierung dazwischen).
  Das beobachtete Muster (Round-Trip klein & über alle 6 Zellen stabil bei 7-12 ms, Offset
  groß & auf < 1 ms stabil bei ≈ -0,248 s) ist die erwartete Signatur einer **echten**
  Uhrendifferenz zwischen Hans' Systemuhr und der PC-Uhr (z. B. fehlender NTP-Sync auf dem
  Pi), nicht eines Messfehlers im Handshake-Code. Kein Code-Fix nötig — die bestehende
  Offset-Korrektur behandelt das bereits richtig (siehe 7.1 Punkt 4). Vor dem produktiven
  Sweep ggf. `timedatectl`/NTP-Sync auf Hans prüfen, wenn ein kleinerer Offset gewünscht ist.
- Falls Rover-Lab-Läufe scheitern: zuerst `test_data_dirs`/Dataset-Export prüfen (Risiko
  #2), erst danach den Szenario-Code selbst verdächtigen — die Konfiguration ist neuer
  als der zugrunde liegende Harness-Code.

---

## 10. Während des Testlaufs entdeckte Befunde (live protokolliert, 2026-07-21)

Diese sind tatsächlich beim echten Deployment auf Hans aufgetreten bzw. beim Sichten der
dabei erzeugten Dateien entdeckt worden — nicht aus reinem Code-Review, sondern verifiziert.

### 10.1 `k`-Mismatch bei allen drei KNN-Szenarien — Config lügt

`config.json` sagt `tinylcm.adaptive_classifier.k: 3`. Tatsächlich zur Laufzeit aktiv ist
**immer `k=5`**, für `scenario_cifar10_knn`, `scenario_imagenette_knn` und
`scenario_rover-lab_knn` gleichermaßen (alle drei geprüft, identisches Muster). Ablauf,
Zeile für Zeile verifiziert:

1. `main.py:900-901` konstruiert `LightweightKNN(k=knn_config['k'], ...)` — hier zählt kurz
   die Config (`k=3`).
2. `main.py:908-915` lädt direkt danach `initial_state/knn_initial_state_objects.json` und
   ruft `classifier.set_state(loaded_state_data['classifier'])`.
3. [`tinylcm_python/knn.py:275`](../../server/Retrainer/on-device-drift-detection-main/tinylcm_python/knn.py)
   `self.k = state.get("k", self.k)` — überschreibt den Konstruktor-Wert mit dem `k` aus
   der geladenen State-Datei.
4. Die State-Datei enthält `k=5`, weil `create_initial_state.py:69` eine **hartcodierte
   Konstante** `KNN_K = 5` definiert, komplett unabhängig von `config.json`, und diese beim
   Bauen des Initial State verwendet (Zeile 507).

**Praktische Konsequenz:** `config.json`s `k` ist für den Klassifikator reine Deko — sie
wird nie wirksam, solange `create_initial_state.py` nicht mit demselben `k` neu ausgeführt
wird. Das betraf schon die komplette historische 1440-Läufe-Kampagne (immer k=5, nicht
k=3 wie die Config suggeriert) — konsistent falsch, aber falsch dokumentiert. Für das Paper
relevant, falls irgendwo "k=3" als Methodik-Detail auftaucht.

**Handlungsbedarf für die neue Session:** Entweder `KNN_K` in allen drei
`create_initial_state.py`-Kopien auf den in `config.json` dokumentierten Wert angleichen
(und Initial States neu erzeugen), oder `config.json`s `k` auf 5 korrigieren, damit Config
und Realität wieder übereinstimmen. Nicht einfach den einen oder anderen Wert ändern, ohne
sich zu entscheiden, welcher der beabsichtigte ist. Sicherer: config direkt einlesen.

### 10.2 Fehlendes automatisches Aufräumen alter Deployments auf dem Pi

`DeploymentManager` in `client/deployment-receiver.py` trackt nur das *aktuell laufende*
Deployment im Prozessspeicher (`self.current_path`) und löscht beim nächsten Deploy nur
genau dieses eine. Über Service-Neustarts hinweg (z. B. nach einem NATS-Ausfall, siehe
10.3) beginnt das Tracking bei `None` neu — alte `deployment_*`-Ordner bleiben für immer
liegen. Bei diesem Testlauf hatten sich so **über 2600 alte Ordner** aus der historischen
Kampagne angesammelt; das Aufräumen allein hat auf der langsamen Pi-Zero-2W-SD-Karte über
6 Minuten gedauert (`rm -rf`, Disk-I/O-gebunden). **Vorschlag (vom Nutzer)**: nach jedem
erfolgreich abgeschlossenen Deployment automatisch alte, nicht mehr benötigte
`deployment_*`-Ordner löschen, statt das nur einmal pro Prozesslebenszeit zu tun. Beim
`--measure-baseline`-Aufräumen (Prompt-Datei) mit erledigen oder als eigenen kleinen Fix.

### 10.3 JetStream-Push-Subscription übersteht keinen Broker-Neustart zuverlässig

Nach einem Docker-Neustart des NATS-Containers (`docker start nats-jetstream`, ohne den
Pi-Service neu zu starten) blieb eine an den Pi gerichtete Nachricht als `num_pending: 1`
im MODELS-Stream hängen, ohne je zugestellt zu werden (`delivered.consumer_seq` blieb bei
`0`), obwohl der Pi-Client laut eigenem Log wieder verbunden war. Erst ein Neustart von
`deployment-receiver.service` auf dem Pi (frische JetStream-Subscription) hat die Nachricht
sofort zugestellt. **Für den produktiven Sweep:** Nach jedem NATS-Neustart (Docker-Neustart,
Absturz, Host-Neustart) den `deployment-receiver`-Service auf dem Pi ebenfalls neu starten
— sich nicht auf automatisches Reconnect/Resubscribe verlassen.

### 10.4 Crash bei relativem `--results-dir`-Pfad

`benchmark_pipeline.py:1724` (`_run_scenario`) ruft
`result_csv.parent.relative_to(PROJECT_ROOT)` auf, um eine Fortschrittszeile zu drucken.
`PROJECT_ROOT` ist immer absolut (Python ≥3.9 garantiert `__file__` als absoluten Pfad),
aber `args.results_dir` (aus `--results-dir`, `type=Path`) bleibt **relativ**, wenn es als
relativer Pfad übergeben wird — genau wie es die `README.md` selbst vorschlägt
(`--results-dir results_run2`). `Path.relative_to()` akzeptiert keine Mischung aus
relativem und absolutem Pfad und wirft `ValueError`, **nachdem das Experiment bereits
erfolgreich abgeschlossen** und das Ergebnis gespeichert wurde — die Exception propagiert
unbehandelt durch `run_pipeline()` und bricht die **gesamte restliche Sweep-Kampagne** ab.
Reproduziert in diesem Testlauf beim ersten erfolgreichen Experiment.
**Workaround (verwendet):** `--results-dir` immer als absoluten Pfad übergeben.
**Echter Fix (für neue Session):** `args.results_dir = args.results_dir.resolve()` direkt
nach dem Parsen in `main()`, oder `os.path.relpath()` statt `Path.relative_to()` an der
betroffenen Stelle verwenden.

### 10.5 Zu entfernen: Konsolen-Ausgabe der Referenzstatistiken bei Initial-State-Erstellung

In jeder Kopie von `create_initial_state.py` (eine pro Szenario) druckt der Block ab
Zeile 628 (`scenario_cifar10_knn`-Beispiel; identisch dupliziert in den anderen Szenarien)
nach dem Erzeugen des Initial State unaufgefordert die vollständigen Referenzstatistiken
und eine "empfohlene KNNDistanceMonitor-Konfiguration" auf die Konsole (Zeilen 629, 649,
657 ff.). Das ist reine Nutzer-Anfrage: **entfernen** — keine funktionale Bedeutung, nur
Log-Rauschen, das bei jedem Sweep-Schritt erneut erscheint.

### 10.6 `initial_state/`-Dateien waren trotz `.gitignore`-Eintrag weiter getrackt

`.gitignore` wurde um `.../initial_state/`-Einträge für alle 9 Szenarien ergänzt, blieb
aber wirkungslos für bereits getrackte Dateien (Git ignoriert `.gitignore` für Pfade, die
schon im Index stehen). Fix in dieser Session angewendet: `git rm -r --cached` für die 6
Szenarien, die tatsächlich getrackte `initial_state/`-Dateien hatten (die 3
Rover-Lab-Szenarien waren nie getrackt). **Nur gestaged, nicht committed** — Dateien
bleiben unangetastet auf der Platte, `git status` zeigt die Entfernung aus dem Index als
bereit zum Commit.

### 10.7 `scenario_cifar10_knn_benchmark` existiert nicht mehr

`README.md` und die ursprüngliche Fassung dieses Testplans referenzierten
`scenario_cifar10_knn_benchmark` — der Ordner heißt inzwischen `scenario_cifar10_knn`
(passend zur produktiven `benchmark_config.json`). Nur `raw_data/` (reiner
Packager-Cache, wird beim nächsten Deploy überschrieben) hat noch den alten Namen. Die
`README.md` ist an dieser Stelle nicht mehr aktuell.

### 10.8 Rover-Lab: hartcodierter 32×32-Dummy-Input crasht `create_initial_state.py` — **live gefixt**

Alle drei Rover-Lab-Szenarien (`scenario_rover-lab_knn`, `_nearest_centroid`, `_kmeans`)
crashten beim allerersten Aufruf von `create_initial_state.py` (ausgelöst durch den
Baseline-Block, bevor überhaupt ein echtes Experiment lief) mit:

```
ValueError: Cannot set tensor: Dimension mismatch. Got 32 but expected 160 for dimension 1 of input 0.
```

Ursache: In allen drei Kopien von `create_initial_state.py` (Zeile 368 bzw. 379) stand
`dummy_rgb = np.zeros((32, 32, 3), dtype=np.uint8)` — hartcodiert von der CIFAR-10-Vorlage
kopiert (die Konsolenausgabe druckte sogar noch "Creating initial KNN state for CIFAR-10"
für den Rover-Lab-Lauf — eindeutiges Copy-Paste-Indiz). Das Rover-Lab-Modell erwartet aber
160×160 (`input_details['shape']`). Dieser Bug hätte **jeden** Rover-Lab-Lauf zum Absturz
gebracht — bestätigt, dass die Szenarien wirklich noch nie erfolgreich deployt wurden.

**Fix (angewendet, alle drei Dateien):**
```python
_in_h, _in_w = int(input_details['shape'][1]), int(input_details['shape'][2])
dummy_rgb = np.zeros((_in_h, _in_w, 3), dtype=np.uint8)
```
Liest die Auflösung jetzt dynamisch aus dem geladenen TFLite-Modell aus, statt sie
hartzucodieren — funktioniert damit unabhängig von der tatsächlichen Szenario-Auflösung.
Syntax aller drei Dateien geprüft (`ast.parse`); funktionale Verifikation durch den
Testlauf selbst (siehe Abschnitt 7.3).

---

## 11. In der Folgesession behobene Befunde (2026-07-21, Fortsetzung)

Alle unten gelisteten Punkte wurden in der auf diesen Testlauf folgenden Session
bearbeitet. Syntax aller geänderten Python-Dateien geprüft (`ast.parse`); kein Zugriff
auf echte Hardware in dieser Session — funktionale Verifikation steht beim nächsten
echten Deployment auf Hans noch aus (insbesondere 10.1/10.2/10.4/7.6).

### 11.1 Befund 10.1 (k-Mismatch) — behoben, KNN + Kmeans; Nearest-Centroid bewusst unverändert

- **KNN-Szenarien** (`scenario_cifar10_knn`, `scenario_imagenette_knn`,
  `scenario_rover-lab_knn`): `create_initial_state.py` liest `KNN_K` jetzt aus
  `config.json`s `tinylcm.adaptive_classifier.k` statt hartcodiert `5`.
- **Kmeans-Szenarien** (`scenario_cifar10_kmeans`, `scenario_imagenette_kmeans`,
  `scenario_rover-lab_kmeans`): gleicher Bug zusätzlich entdeckt und behoben — betraf
  `KMeansPrototypeClassifier` genauso (via `set_state()` in `tinylcm_python/knn.py`).
  Zusätzlich `KMEANS_M_PROTOTYPES` aus `config.json`s `m_prototypes` gelesen (gleiche
  Bug-Klasse, war bisher nur zufällig konsistent).
- **Nearest-Centroid-Szenarien**: **kein Bug** — `NearestCentroidClassifier.fit()`
  setzt `k` immer selbst auf die Klassenanzahl, unabhängig von Config oder Konstante
  (siehe `tinylcm_python/prototype_classifier.py`). `KNN_K`/`KMEANS_M_PROTOTYPES` dort
  auskommentiert statt aktiviert (siehe 11.5).
- `"knn_k"`-Feld in `knn_reference_stats.json` liest jetzt überall `knn.k` (den
  tatsächlichen Laufzeit-Wert) statt der Konstante — behebt die falsche `"knn_k": 5`
  Angabe für Nearest-Centroid, wo der reale k-Wert die Klassenanzahl ist.
- **Offen:** Initial States müssen beim nächsten Deploy automatisch neu gebaut werden
  (passiert bei jedem `create_initial_state()`-Aufruf ohnehin), kein manueller Schritt
  nötig — aber noch nicht auf Hans verifiziert.

### 11.2 Befund 10.2 (kein Auto-Cleanup alter Deployments) — behoben

`client/deployment-receiver.py`: neue Funktion `_cleanup_old_deployments()`, aufgerufen
nach jedem erfolgreichen `start_new()` in `process_deployment_zip()`. Löscht alle
`deployment_*`-Ordner außer dem gerade gestarteten — ein zustandsloser
Verzeichnis-Scan, der auch Service-Neustarts übersteht (im Gegensatz zum bisherigen
`DeploymentManager.current_path`-Tracking, das bei jedem Prozess-Neustart bei `None`
beginnt).

**Nachtrag (nach Rückfrage zur Wechselwirkung mit 10.3):** Ohne Schutzmaßnahme hätte
das aggressive Löschen einen Ordner, der durch einen Service-Neustart nach einem
NATS-Neustart (siehe 10.3) verwaist ist (Logs nie übertragen, da das In-Memory-Tracking
verloren ging), beim nächsten erfolgreichen Deploy unwiderruflich entfernt — ohne
Chance auf nachträgliche Fehlersuche. Ergänzt: `DEPLOYMENT_CLEANUP_GRACE_S = 3600`
(1 h Gnadenfrist über `entry.stat().st_mtime`) — ein Ordner wird erst gelöscht, wenn er
seit mind. 1 h nicht mehr angefasst wurde — plus ein `logging.info()` mit Ordnername
und Alter unmittelbar vor jedem `shutil.rmtree()`, damit auch nach der Löschung ein
Eintrag in `journalctl -u deployment-receiver` bleibt.

### 11.3 Befund 10.3 (JetStream-Subscription übersteht Broker-Neustart nicht) — nur Runbook, kein Code-Fix

Bewusste Entscheidung: kein Code-Änderung (Risiko kurz vor dem produktiven Sweep),
stattdessen zwei neue Doku-Stellen in `README.md`:
- Troubleshooting-Tabelle: neue Zeile "Deployment message stuck... after a NATS broker
  restart".
- Abschnitt "Raspberry Pi Receiver Over SSH": Hinweis, `deployment-receiver.service`
  nach *jedem* NATS-Neustart neu zu starten, nicht nur nach Code-Updates.

### 11.4 Befund 10.4 (Crash bei relativem `--results-dir`) — behoben

`benchmark_pipeline.py::main()`: `args.results_dir = args.results_dir.resolve()`
direkt nach `parser.parse_args()` — macht den Pfad immer absolut, behebt den
`Path.relative_to()`-Crash an der Wurzel statt nur an der einen betroffenen Zeile.

### 11.5 Befund 10.5 (Konsolen-Rauschen bei Initial-State-Erstellung) — behoben

Der unaufgefordert gedruckte "Empfohlene KNNDistanceMonitor-Konfiguration"-Block
(Berechnung von `suggested_delta`/`suggested_lambda`/`stable_classes` + der große
JSON-Beispiel-Print) aus allen 9 Kopien von `create_initial_state.py` entfernt. Die
eigentliche Berechnung und Speicherung von `knn_reference_stats.json` (potenziell für
künftige `KNNDistanceMonitor`-Konfiguration nützlich) bleibt unverändert erhalten —
nur die Konsolenausgabe ist weg.

Im selben Aufwasch (nicht explizit in 10.5 gefordert, aber zur Portierung des k-Fixes
angefallen): in den 6 Kopien mit `CLASSIFIER_TYPE`-Switch (`_nearest_centroid`/
`_kmeans`-Szenarien) wurden die jeweils zwei toten Klassifikator-Zweige, tote Imports
und die nie aufgerufene Funktion `calculate_knn_distance_statistics()` (Duplikat von
`..._from_knn`) auskommentiert statt gelöscht — Kommentare erklären jeweils, warum
(Portierbarkeit zwischen den 9 Szenario-Kopien).

### 11.6 Befund 10.6 (`initial_state/`-Dateien trotz `.gitignore` getrackt) — bereits erledigt, keine Aktion nötig

Geprüft: `git ls-files` zeigt keine getrackten Dateien mehr unter irgendeinem
`initial_state/`-Pfad, `.gitignore` enthält alle 9 Einträge, `git status` zeigt keine
offenen Staging-Reste. Der Commit `8900ebd` ("Initial test run") hat die in Abschnitt
10.6 beschriebene `git rm --cached`-Änderung offenbar bereits mit committet.

### 11.7 Befund 10.7 (stale `scenario_cifar10_knn_benchmark`-Referenzen) — behoben, Scope erweitert

Alle Vorkommen von `scenario_cifar10_knn_benchmark` in `README.md` auf
`scenario_cifar10_knn` korrigiert (Scenario-List-Beispiel, Setup-Anleitung,
Mismatch-Beispiel in Schritt 5, Result-Layout-Baum). Zusätzlich (nach Rückfrage)
die Sektion "Current Default Config" komplett auf die tatsächlichen 9 Szenarien
aktualisiert — sie listete vorher nur 6 und darunter noch den falschen Namen; die
3 rover-lab-Szenarien fehlten komplett.

### 11.8 Befund 7.1 (Uhren-Offset ≈ -0,248 s) — Ursache untersucht, kein Code-Bug; neuer Audit-Check ergänzt

**Untersuchungsergebnis:** Die in Abschnitt 9 geäußerte Vermutung (blockierende
Operation vor der Zeitstempel-Erfassung in `handle_time_sync()`) trifft **nicht** zu.
Sowohl `handle_time_sync()` (Pi-Seite, `client/deployment-receiver.py:354-359`) als
auch `_measure_clock_offset()` (PC-Seite, `benchmark_pipeline.py:915-943`) erfassen
ihre Zeitstempel bereits unmittelbar, ohne Awaits/Blockierung dazwischen — verifiziert
durch Code-Lesen, nicht durch erneuten Hardware-Test. Das beobachtete Muster (Round-Trip
klein und über alle 6 Zellen stabil bei 7-12 ms, Offset groß und auf < 1 ms stabil bei
≈ -0,248 s) ist die erwartete Signatur einer **echten** Uhrendifferenz zwischen Hans'
Systemuhr und der PC-Uhr, nicht eines Messfehlers im Handshake-Code. Siehe auch den
Nachtrag in Abschnitt 9 sowie den entsprechenden Nachtrag in
`MESSMETHODIK_ZEITSYNCHRONISATION.md` §5 Punkt 7.

**Nutzerfrage dazu:** Der Handshake läuft bereits vor **jedem einzelnen** Deploy
(jedes Experiment und jeder der 3 Baseline-Runs pro Block, nicht nur einmal global) —
das war schon vor dieser Session so implementiert (siehe Abschnitt 1.2). Was fehlte,
war ein automatisierter Konsistenz-Check des gemessenen Werts (in
`MESSMETHODIK_ZEITSYNCHRONISATION.md` §5 Punkt 7 nur als Konzept beschrieben, nie
implementiert) — jetzt ergänzt:

**Fix:** `analysis/verify_power_coverage.py` bekommt einen neuen Check 7: liest
`clock_offset.offset_s`/`round_trip_s` aus der Metadatei jedes Laufs (Experiment- und
Baseline-Run-Ordner gleichermaßen, dank 7.6-Fix auch für Baseline-Runs verfügbar),
schreibt beide Werte als neue Spalten (`clock_offset_s`, `clock_offset_round_trip_s`)
in `power_coverage_audit.csv` und meldet `WARN`, wenn `|offset_s|` den bestehenden
`--offset-warn`-Schwellenwert (Default 1,0 s, bisher nur für den unzuverlässigen
Kreuzkorrelations-Check verwendet) überschreitet. Die aggregierte "Paper numbers"-
Ausgabe zeigt zusätzlich Median/Maximum des gemessenen Offsets über den gesamten
Sweep. Bewusst ein fixer Sanity-Schwellenwert, keine automatisierte Prüfung gegen die
historische kausale Schranke aus Abschnitt 5.1 (andere Hardware/Kampagne, laut
Dokument selbst nicht 1:1 übertragbar) — dieser Vergleich bleibt manuelle
Interpretation wie in diesem Testlauf.

### 11.9 Befund 7.6 (fehlende Timestamps für Baseline-Runs) — behoben

`run_baseline_block()` in `benchmark_pipeline.py` schreibt jetzt pro Run eine eigene
`experiment_metadata.json` in den jeweiligen Run-Unterordner (`baseline_XXX_runK/`,
wo auch `power_*.csv`/`resource_usage_*.csv` bereits liegen) mit `started_at`,
`result_received_at` und `clock_offset` — spiegelt exakt den Experiment-Pfad.
`verify_power_coverage.py::read_metadata()`/`discover_experiments()` brauchten dafür
keine Änderung, da sie bereits nach genau diesem Dateinamen suchen. Damit greift der
kausale Containment-Check (Punkt 2) jetzt auch für alle Baseline-Runs, nicht nur für
echte Experimente.

### 11.10 Zusatzbefund: `power.rate` in der produktiven Config war stale (5.0 statt 8.0)

Beim Durchsehen von Abschnitt 8.2 aufgefallen: Die produktive `benchmark_config.json`
hatte weiterhin `"power": {"rate": 5.0}` — obwohl `test_config_validation.json` (dieser
Testlauf) bereits korrekt `8.0` verwendete und `dsp3005_reader.py`s `DEFAULT_RATE` seit
der in Abschnitt 1.6 dokumentierten Sampling-Rate-Verbesserung `8.0` ist. Da
`power_cfg.get("rate", 8.0)` den expliziten Config-Wert immer vor dem Fallback zieht,
hätte der produktive Sweep sonst stillschweigend wieder mit der alten, gröberen 5-Hz-Rate
gelaufen. **Fix (angewendet, nach Rückfrage):** `benchmark_config.json`s `power.rate` auf
`8.0` gesetzt. `power.port` wurde parallel vom Nutzer selbst von `COM3` auf `COM5`
aktualisiert (echter Port von Hans, siehe Abschnitt 4.7/8.2) — nicht von dieser Session
verändert, hier nur zur Vollständigkeit vermerkt.

### 11.11 Grobe Zeitschätzung für den produktiven Sweep (informativ, kein Fix)

Auf Nutzeranfrage aus den echten `started_at`/`result_received_at`-Zeitstempeln des
Testlaufs (`results_validation/`, `test_images_per_class=5`) berechnet: fixer
Overhead (Deploy + Pi-Setup + Log-Merge, aus `total_cycle_s − inference_window_s`)
getrennt von der Inferenzzeit/Bild, dann auf die produktive Config
(`test_images_per_class=20`, 12 `samples_per_class`, 2 PCA-Modi, 10 Seeds ⇒ 240
Experimente + 72 Baseline-Runs = 312 Deployments je Szenario, 2808 insgesamt über alle
9 Szenarien) hochskaliert:

| Datensatz | Testbilder/Lauf | ~Zeit/Deployment | Familie (3 Klassifikatoren) |
|---|---|---|---|
| CIFAR-10 | 200 | ~61 s | ~15,9 h |
| Imagenette | 200 | ~134 s | ~34,8 h |
| Rover-Lab | 40 | ~31 s | ~8,0 h |

**Ergebnis: ≈ 58,6 h ≈ 2,4 Tage** reiner Deploy-Zyklen — ein optimistischer Boden, kein
belastbarer Endwert. Nicht mit eingerechnet: `create_initial_state()` läuft PC-seitig
**vor** `started_at` (in den gemessenen Zykluszeiten also gar nicht enthalten) und
skaliert mit `samples_per_class` (bis 100/Klasse); außerdem keine Zeit für
Timeouts/Retries, Chunked-Transfer bei großen Imagenette-Deployments, oder die laut
10.3 zu erwartenden gelegentlichen NATS-/Service-Neustarts. Realistische Erwartung:
**3–4 Tage unbeaufsichtigter Lauf statt 2,4 Tage**.
