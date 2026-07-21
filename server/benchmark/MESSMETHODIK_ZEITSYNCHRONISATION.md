# Zeitliche Zuordnung der Energiemessung: Methodik und Validität

Dieses Dokument erklärt, wie die Inferenz-Energiemessung im Benchmark zeitlich
funktioniert: welche Uhren beteiligt sind, wie das Inferenzfenster exakt
bestimmt wird, warum der fehlende explizite Uhrenabgleich die Messungen unter
den unten genannten — prüfbaren — Bedingungen nicht maßgeblich verfälscht, und
wie wir das nachweisen.

> **Update (2026-07-21):** Seit dem Result-Attribution-Fix vom 2026-07-20 (siehe
> [`README.md` § „Result Attribution & Energy-Data Integrity"](README.md)) wird
> jede Ergebnis-ZIP nicht mehr anhand ihrer *Ankunftsreihenfolge*, sondern
> anhand einer im NATS-Publish mitgesendeten **Experiment-ID** (`X-Experiment-Id`
> / `X-Scenario-Name`) geroutet. Das entkoppelt die in Abschnitt 5 beschriebene
> „Zuordnung" strukturell vom hier behandelten Uhren-Thema — siehe Abschnitt 5
> für Details. Der Audit in Abschnitt 5.1 stammt aus der Messkampagne **vor**
> diesem Fix und vor der Baseline-/EDP-Neuausrichtung (Sampling-Rate, warme
> Baseline); die gesamte Kampagne wird neu erfasst, die alten Zahlen dienen nur
> noch als historische Referenz (als solche markiert).

## 1. Messaufbau: zwei Uhren, ein Messgerät ohne eigene Uhr

| Komponente | Rolle | Zeitquelle |
|---|---|---|
| Voltcraft DSP-3005 | misst Spannung/Strom der Pi-Versorgung | **keine** — liefert nur V/I per SCPI |
| Benchmark-PC | pollt das DSP-3005 mit 8 Hz (Ziel; Default seit der VOLT?/CURR?-Optimierung, vorher 5 Hz), stempelt jede Zeile | PC-Systemuhr (`datetime.now()`, `dsp3005_reader.py:111`) |
| Raspberry Pi | loggt CPU/RAM mit 5 Hz während der Inferenz | Pi-Systemuhr (`time.time()`, `resource_usage_*.csv`) |

Die Leistungs-Zeitreihe (`power_*.csv`) lebt also vollständig auf der PC-Uhr,
das Inferenzfenster vollständig auf der Pi-Uhr. Ein expliziter
Synchronisationsschritt (z. B. NTP-Handshake) findet nicht statt; beide
Betriebssysteme synchronisieren ihre Uhren standardmäßig selbst per NTP
(Raspberry Pi OS: `systemd-timesyncd`, Windows: `time.windows.com`).

## 2. Exakter Ablauf der Inferenz-Energiemessung

Pro Experiment führt `benchmark_pipeline.py` folgende Sequenz aus
(`_subscribe_deploy_collect`, ab Zeile 888):

```
PC:  PowerLogger.start()          ← Messung beginnt VOR dem Deployment
PC:  Deployment an den Pi (NATS), config.json trägt experiment_id
Pi:  Deployment empfangen, entpacken, Python-Start
Pi:  Setup: TFLite-Modell laden (eager), PCA laden, KNN-State laden
Pi:  ResourceCSVLogger.start()    ← erste Zeile sofort, dann alle 200 ms
Pi:  Inferenz-Schleife über alle Testbilder
Pi:  ResourceCSVLogger.stop()     ← letzte Zeile ≤ 0,2 s nach Schleifenende
Pi:  Merge der Logs, Ergebnis-ZIP mit X-Experiment-Id/X-Scenario-Name-Headern
     über NATS an den PC (client/transmitter.py)
PC:  handle_msg() prüft den Tag gegen die erwartete experiment_id (Zeile 941ff.);
     nur bei Übereinstimmung gilt die ZIP als DAS Ergebnis dieses Deploys —
     eine falsch getaggte ZIP landet im Ordner ihrer eigenen ID, nie hier
PC:  Ergebnis empfangen → PowerLogger.stop()
```

Die Tag-Prüfung ist rein eine **Zuordnungs**-Garantie (welches Experiment gehört
zu welcher ZIP) und arbeitet unabhängig von beiden Uhren — sie ersetzt keine
Zeitsynchronisation, sondern macht sie für die Fensterbestimmung in diesem
Dokument irrelevant: Ab hier ist definitionsgemäß sichergestellt, dass die
folgenden Zeitstempel tatsächlich zum selben Experiment gehören.

Dieselbe Sequenz durchlaufen auch die neuen **Baseline-Blöcke**
(`run_baseline_block()`, siehe unten): je `n=3` Deployments mit
`application.baseline_mode=true` vor der ersten und nach jeder
`samples_per_class`-Zelle, mit identischer ID-Tag-Logik und identischer
Fensterbestimmung (Abschnitt „Referenzen" unten) — nur ohne Klassifikator-Aufruf.
Ihre `recorded_at`-Zeitstempel stammen ebenfalls von der PC-Uhr, genau wie
`started_at`/`result_received_at` der echten Experimente — der Abgleich „welcher
Baseline-Block gehört zu welcher Zelle" (`analyze_benchmark.py::
_local_baseline_power_w`) vergleicht also ausschließlich PC-Zeitstempel
untereinander und ist von der hier behandelten PC↔Pi-Uhrenfrage unberührt.

Das **Inferenzfenster** ist definiert als `[t0, t1] = [min, max]` der
`timestamp_unix`-Werte des Pi-Ressourcen-Logs (`_inference_window`,
`benchmark_pipeline.py:1127`).
Da der Ressourcen-Logger erst nach dem vollständigen Setup startet (Kommentar
im Szenario-Code: *„Start resource logging only when the actual processing
starts"*) und im `finally` unmittelbar nach der Schleife stoppt, umschließt
das Fenster die Inferenzphase auf ±1–2 Samples (±0,2–0,4 s) genau — beide
Kanten stammen von **derselben Uhr** wie die Inferenz selbst.

Die Energie ergibt sich als Trapezintegral der Leistungs-Samples innerhalb
des Fensters (`_integrate`, `benchmark_pipeline.py:1143`). Ausgewiesen werden
`full_window` (gesamter Trace inkl. Deployment) und `inference_window`
(nur Inferenzphase); alle Pro-Bild- und Netto-Metriken beziehen sich auf
letzteres.

**Enthalten** im Fenster: Feature-Extraktion, Klassifikation,
Drift-Detektion, Logging sowie das konfigurierte Inter-Frame-Delay
(200 ms/Bild). **Nicht enthalten:** Deployment-Transfer, Modell-Laden,
Log-Merge, Ergebnis-Übertragung. `energy_per_image_J` ist damit ein
Fensterdurchschnitt der Verarbeitungsschleife, kein reiner Forward-Pass;
`net_energy_per_image_J` (abzüglich separat gemessener Idle-Baseline) ist
die belastbarere Absolutgröße.

## 3. Die Synchronisationsfrage — und ihr Fehlermodell

> **Update (2026-07-21):** Der Uhrenversatz δ wird seit dieser Änderung nicht
> mehr nur indirekt über die kausale Schranke (Abschnitt 5, Punkt 2)
> eingegrenzt, sondern **pro Experiment und pro Baseline-Block aktiv
> gemessen**: ein NTP-artiger Round-Trip-Handshake über NATS Core
> (`time.sync.<device_id>`, sieben Pings, kleinster Round-Trip gewinnt — siehe
> `PLAN_ZEIT_HANDSHAKE.md`) liefert einen Messwert `offset_s`, der in
> `experiment_metadata.json`/`baseline_summary.json` unter dem Schlüssel
> `"clock_offset"` abgelegt wird. `analyze_benchmark.py::
> _offset_corrected_power_metrics` verschiebt das Inferenzfenster damit aktiv
> (`T_pc_equiv = T_pi − offset_s`) und berichtet `avg_power_W_offset_corrected`
> / `net_avg_power_W_offset_corrected` / `EDP_offset_corrected` zusätzlich zu
> den unkorrigierten Größen. Die folgende Fehlermodell-Herleitung bleibt als
> Erklärung *warum* ein Restversatz überhaupt relevant ist gültig, ist aber ab
> jetzt keine reine Plausibilitätsannahme mehr, sondern durch einen Messwert
> ersetzbar; für Daten aus **vor** dieser Änderung (kein `clock_offset` in der
> Metadatei) gilt sie unverändert als einzige verfügbare Fehlerabschätzung.

Die einzige Stelle, an der beide Uhren übereinstimmen müssen, ist die
Übertragung der Fensterkanten (Pi-Zeit) auf die Leistungs-Zeitreihe
(PC-Zeit). Ein Uhrenversatz δ verschiebt das Integrationsfenster um δ auf
dem Trace. Der dadurch entstehende Energiefehler wirkt nur an den beiden
Fensterkanten:

```
ΔE ≈ δ · |P_Inferenz − P_Rand|  pro Kante
relativer Fehler ≈ 2δ · ΔP / (P̄ · T_Fenster)
```

Mit typischen Werten (ΔP ≈ 1 W zwischen Inferenz- und Idle-Leistung,
P̄ ≈ 3–4 W, Fensterlänge T im Minutenbereich) gilt:

- **δ ≈ 50 ms** (NTP-typisch im LAN): Fehler in der Größenordnung 0,01 % — vernachlässigbar.
- **δ = 1 s** (stark degradierte Synchronisation): Fehler < 1 %.
- **δ ≫ Sekunden** (z. B. Pi ohne NTP nach Boot; der Pi hat keine
  batteriegepufferte Echtzeituhr): Fenster und Trace passen nicht mehr
  zusammen — dieser Fall muss ausgeschlossen werden (Abschnitt 5).

Wichtig für die Interpretation der Studie: Die zentralen Aussagen des
Benchmarks sind **relative Vergleiche** zwischen Konfigurationen (PCA-Varianten,
Stützstellenzahl) auf identischem Aufbau. Ein über alle Experimente
konstanter Restversatz wirkt als Common-Mode-Fehler und kürzt sich in diesen
Vergleichen heraus; er beträfe ausschließlich die Absolutwerte.

## 4. Warum die Messung die Inferenz garantiert vollständig abdeckt

Unabhängig von jeder Uhrensynchronisation gilt eine **Kausalkette**
(Happened-before-Ordnung): Der Pi kann erst rechnen, nachdem das Deployment
angekommen ist — und das wird erst versendet, nachdem die Leistungsmessung
läuft. Umgekehrt stoppt die Messung erst, nachdem das Ergebnis-ZIP
eingetroffen ist, das der Pi erst nach Abschluss der Inferenz abschickt.
Die physische Inferenzphase liegt daher **konstruktionsbedingt** immer
innerhalb des aufgezeichneten Leistungs-Traces, mit Sicherheitsrändern von
mehreren Sekunden vorn (Deploy-Dauer) und hinten (Merge + Transfer). Ein
Uhrenfehler kann folglich nicht dazu führen, dass Messdaten fehlen — nur
dazu, dass das *berechnete* Fenster falsch auf dem Trace platziert wird.
Und genau das ist nachträglich prüfbar.

## 5. Nachweis statt Annahme: das Audit

Das Skript `analysis/verify_power_coverage.py` prüft nachträglich für jedes
Experiment aus den ohnehin archivierten Dateien:

1. **Ränder:** Der Trace beginnt vor `t0` und endet nach `t1` (Soll: Sekunden Reserve).
2. **Kausale Einbettung:** Aus den PC-Ereignissen `started_at` (Deploy) und
   `result_received_at` (ZIP-Empfang) folgt die beweisbare Schranke
   `t1 − result_received_at ≤ δ ≤ t0 − started_at`. Ein leeres Intervall oder
   eines, das 0 ausschließt, belegt eine Uhreninkonsistenz — ganz ohne
   NTP-Vertrauen.
3. **Lückenlosigkeit:** Abstände aufeinanderfolgender Samples (Soll 0,2 s) auf
   beiden Seiten — erkennt NTP-Sprünge auf dem Pi mitten im Lauf ebenso wie
   einen abgestürzten Leistungs-Logger.
4. **Zuordnung:** genau eine Ressourcen-CSV, `run_id` konsistent zum Ergebnis.
   Seit dem Result-Attribution-Fix (2026-07-20) ist das keine Wahrscheinlichkeits-,
   sondern eine Baugarantie mehr: `handle_msg()` akzeptiert eine ZIP nur bei
   passendem `X-Experiment-Id`-Tag (siehe Abschnitt 2). Dieser Audit-Punkt prüft
   also ab jetzt lediglich die Konsistenz einer bereits strukturell garantierten
   Zuordnung, nicht mehr deren Herstellung.
5. **Uhrenfreie Offset-Schätzung:** Kreuzkorrelation der Pi-CPU-Last mit der
   PC-Leistungskurve; der Korrelations-Peak liefert den Restversatz δ̂ direkt
   aus den Signalen. **Auf den realen Messdaten nicht aussagekräftig** — siehe
   Abschnitt 5.1.
6. **Sensitivität:** Verschiebung des Fensters um ±2 s und Neuberechnung der
   Energie beziffert den maximal möglichen Randeffekt pro Lauf.
7. **Gemessener Handshake-Offset gegen Sanity-Schwelle (implementiert
   2026-07-21, Folgesession):** `verify_power_coverage.py` liest jetzt aktiv
   `clock_offset.offset_s`/`round_trip_s` aus `experiment_metadata.json` (bzw.
   der pro-Run-Metadatei jedes Baseline-Laufs, siehe Abschnitt 2) und meldet
   `WARN`, wenn `|offset_s|` den Schwellenwert `--offset-warn` (Default 1,0 s,
   siehe Fehlermodell in Abschnitt 3) überschreitet — unabhängig vom kausalen
   Containment-Check (Punkt 2), der auf `started_at`/`result_received_at`
   basiert statt auf der Handschlag-Messung selbst. Beide Spalten
   (`clock_offset_s`, `clock_offset_round_trip_s`) landen zusätzlich in
   `power_coverage_audit.csv`, und die aggregierte "Paper numbers"-Ausgabe
   zeigt Median/Maximum über den gesamten Sweep. Das ist bewusst ein fixer
   Sanity-Schwellenwert, **keine** automatisierte Prüfung gegen die
   historische kausale Schranke δ ∈ [−0,19 s, +3,73 s] aus Abschnitt 5.1 — die
   stammt von anderer Hardware/Kampagne und ist laut eigener Einschränkung dort
   nicht 1:1 übertragbar; ein Vergleich gegen sie bleibt manuelle Interpretation
   (siehe 7.1 im Testplan für ein Beispiel, wie das im Einzelfall aussieht).

Validierung des Audits an synthetischen Testdaten: Ein künstlich eingebauter
Uhrenversatz von +3 s wurde per Kreuzkorrelation auf die Sekunde genau
erkannt (r = 0,98), ein simulierter NTP-Sprung von 300 s mitten im Lauf
schlug in vier unabhängigen Checks fehl; ein sauberer Lauf passierte mit
δ̂ = 0,0 s und einer Fenster-Sensitivität von 0,5 %. **Diese Zahlen stammen
ausschließlich aus dem synthetischen Testfall und sind nicht auf die reale
Messkampagne übertragbar** (Abschnitt 5.1).

## 5.1 Ergebnis des Audits über die reale Messkampagne (1440 Läufe) — historisch

> **Diese Zahlen sind archiviert, nicht aktuell.** Ausgeführt am 2026-07-15 —
> **vor** dem Result-Attribution-Fix (2026-07-20, Abschnitt 2/5) und vor der
> Baseline-/EDP-Neuausrichtung (warme, lokale Baseline statt kalter globaler
> Baseline; 8 Hz statt 5 Hz Power-Sampling — siehe `dsp3005_reader.py`). Die
> gesamte Messkampagne wird auf dieser neuen Grundlage komplett neu erfasst;
> die alten `results/**`-Daten dienen nur noch zur Orientierung. Ein frischer
> Audit-Lauf über die neue Kampagne ist vorgesehen, sobald sie abgeschlossen
> ist — mit erwartbar **0 statt 3 FAIL** (Ursache strukturell behoben, siehe
> unten) und angepassten Lücken-Sollwerten (0,125 s statt 0,2 s bei 8 Hz).
> Bis dahin dokumentiert dieser Abschnitt nur, *dass* die Methodik sich schon
> einmal bewährt hat — nicht die aktuellen Kennzahlen.

Ausgeführt am 2026-07-15 über `results/`; Rohergebnis pro Lauf in
`results/analysis/power_coverage_audit.csv`. **258 PASS, 1179 WARN, 3 FAIL.**
Die WARN-Fälle sind fast ausschließlich die Punkte 5 und 6 und **keine
Messfehler** — siehe unten.

**Belastbar (Punkte 1–4):**
- Kein einziger Lauf ohne vollständige Fensterüberdeckung: Vormargin ≥ 3,56 s,
  Nachmargin ≥ 0,08 s, ≥ 96,5 % der erwarteten Leistungs-Samples im Fenster.
- Keine Uhrensprünge: größte Pi-Sampling-Lücke 0,73 s (nominal 0,2 s), alle
  Zeitstempel monoton.
- **Kausale Schranke: bei allen 1437 gültigen Läufen enthält das Intervall die
  Null.** Der Schnitt über alle Läufe ergibt δ ∈ [−0,19 s, +3,73 s] — ohne
  jedes Vertrauen in NTP. Das ist die tragende Aussage dieses Dokuments.

**Nicht belastbar (Punkt 5 — Kreuzkorrelation):** Die CPU-Last ist während der
Inferenz nahezu gesättigt und damit strukturarm; es gibt fast nichts zu
korrelieren. Der Peak liegt im Median bei r = 0,40, nur 1 von 1437 Läufen
erreicht r > 0,7, und die geschätzten Versätze streuen mit σ = 9,0 s um einen
Median von +0,40 s bei annähernd 50/50 verteilten Vorzeichen. Vier Läufe
laufen exakt gegen die ±30-s-Suchgrenze. Das ist **Schätzrauschen, kein
Uhrenversatz**. Die vom Skript ausgegebene Zeile „residual clock offset:
median 1,60 s" darf **nicht** zitiert werden.

**Sensitivität (Punkt 6) — real deutlich über „sub-Prozent":** Median 1,66 %,
p95 4,52 %, Maximum 5,42 %; 55 % der Läufe liegen über 1 %. Das ist echte
Physik und kein Artefakt: Der Randeffekt skaliert mit 1/T, wie die Formel in
Abschnitt 3 vorhersagt (Korrelation Fensterlänge ↔ Sensitivität = −0,95).

**Die Trennlinie ist der Datensatz, nicht der Sample-Count.** Die Fensterlänge
ergibt sich aus 200 Testbildern × Zeit/Bild und ist innerhalb eines Datensatzes
praktisch konstant — CIFAR-10 liegt bei 49,4 s für *jedes* `spc` von 2 bis 100,
Imagenette bei 116,1 s. Entsprechend:
- **CIFAR-10** (Fenster ~49 s): Median 3,67 % — betrifft alle Konfigurationen gleichermaßen.
- **Imagenette** (Fenster ~116 s): Median 0,63 %.

**Verstärkung bei `net_energy_per_image_J`:** Die Netto-Größe zieht die
Idle-Baseline ab und beträgt damit nur ~23 % (CIFAR-10) bzw. ~45 % (Imagenette)
der Bruttoenergie. Eine Fensterverschiebung ändert die Energie um einen
*absoluten* Betrag ΔE, die Fensterdauer bleibt gleich — der relative Fehler auf
die Netto-Größe ist daher um den Faktor Brutto/Netto verstärkt (≈ 4,3×
CIFAR-10, ≈ 2,2× Imagenette).

Wichtig für die Einordnung: Die ±2 s sind ein **Was-wäre-wenn**, keine
Fehlerangabe. Sie beziffern den Fehler, *falls* der Versatz 2 s betrüge; die
kausale Schranke oben begrenzt ihn auf ≤ 3,73 s und ist mit 0 verträglich.
Bei realistischem NTP-Versatz (Millisekunden) ist der Effekt vernachlässigbar.
Relative Vergleiche zwischen Konfigurationen sind ohnehin unberührt
(Common-Mode) — betroffen sind ausschließlich Absolutwerte, in erster Linie
`net_energy_per_image_J` und `net_avg_power_W`.

**3 FAIL — und zwar kein Uhrenproblem.** `exp_004_pca95var_spc2_s789`,
`exp_039_pca95var_spc20_s1618` und `exp_125_nopca_spc2_s1337` haben je eine
**byte-identische Kopie des Ergebnisses des Vorgänger-Experiments** erhalten
(gleiche `run_id`, per md5 verifiziert). Der Nachweis läuft auf **einer**
Uhr und braucht keine Synchronisation: Der PC sah den kompletten Zyklus
Deploy→Ergebnis in 4,3–4,9 s, während das zurückgelieferte Ressourcen-Log
eine 49–114 s lange Inferenz beschreibt. Es handelt sich also um eine
fehlerhafte **Ergebnis-Zuordnung** (stale Message trotz `DeliverPolicy.NEW`),
nicht um Uhrendrift. Die Läufe sind in `analysis/analyze_benchmark.py` über
`EXCLUDED_EXPERIMENTS` aus allen Aggregaten entfernt; die betroffenen Zellen
behalten 9 von 10 Seeds. Es hat sich nicht fortgepflanzt — der jeweils
nächste Lauf ist sauber.

**Diese drei FAILs sind genau der Fehler, den der Result-Attribution-Fix
(Abschnitt 2/5, Punkt 4) strukturell verhindert:** Die stale Nachricht hätte
mit dem heutigen `handle_msg()` nicht mehr das aktuell erwartete Experiment
„getroffen" — sie wäre anhand ihres eigenen `X-Experiment-Id`-Tags in ihren
eigenen (bereits abgeschlossenen) Ordner geroutet worden, ohne den aktuellen
Lauf zu berühren. In der neuen Kampagne ist dieser Fehlermodus nicht mehr
möglich; `EXCLUDED_EXPERIMENTS` sollte für neu erfasste Daten leer bleiben.

**Die Validitätsaussage ist damit bedingt und prüfbar.** Nach dem realen
Audit-Lauf (Abschnitt 5.1) ist belegt — nicht angenommen —, dass
(a) jeder Trace sein Fenster mit Reserve überdeckt, (b) kein Uhrensprung
auftrat und (c) der Restversatz kausal auf δ ∈ [−0,19 s, +3,73 s] beschränkt
und mit 0 verträglich ist. Punkt (d) ist gegenüber der ursprünglichen
Fassung zu korrigieren: Eine Fensterverschiebung um ±2 s ändert die Energie
**nicht** durchgängig im Sub-Prozent-Bereich, sondern um median 1,66 % und
maximal 5,42 %, konzentriert auf die kurzen Läufe (1/T-Effekt). Für die drei
Läufe, die das Audit als FAIL markiert, gilt unverändert: verwerfen oder
wiederholen — sie sind verworfen (Abschnitt 5.1).

## 6. Grenzen und Ausblick

### Was durch die letzten Änderungen bereits gelöst ist

- **Ergebnis-Zuordnung** (welche ZIP gehört zu welchem Experiment) ist seit
  2026-07-20 keine Wahrscheinlichkeitsfrage mehr, sondern strukturell per
  ID-Tag garantiert (Abschnitt 2/5). Das war historisch der einzige real
  beobachtete Fehlerfall (3 von 1440 Läufen, Abschnitt 5.1) — dieser
  Fehlermodus ist jetzt ausgeschlossen, nicht nur seltener geworden.
- Die **Baseline-Zuordnung** (lokale, zeitlich benachbarte Warm-Baseline pro
  Zelle statt einer einzigen globalen Kalt-Baseline) vergleicht ausschließlich
  PC-seitige Zeitstempel (`recorded_at` vs. `started_at`/`result_received_at`)
  — hier existiert keine PC↔Pi-Uhrenfrage, nur eine Sequenzialitätsannahme
  (siehe unten).
- **Die eigentliche Uhrenfrage (PC ↔ Pi) ist seit 2026-07-21 gelöst, nicht
  mehr nur eingegrenzt.** Ein NTP-artiger Round-Trip-Handshake über NATS Core
  (`time.sync.<device_id>`, Pi-seitig `client/deployment-receiver.py::
  handle_time_sync`, PC-seitig `benchmark_pipeline.py::
  _measure_clock_offset`) misst `offset_s` aktiv vor jedem Deploy und vor
  jedem Baseline-Block, statt sich auf die impliziten NTP-Voreinstellungen
  der Betriebssysteme zu verlassen. `analyze_benchmark.py::
  _offset_corrected_power_metrics` wendet die Korrektur auf das
  Inferenzfenster an (siehe Abschnitt 3). Der kausale Audit (Abschnitt 5,
  Punkt 2) bleibt als Konsistenz-Check gegen den Messwert erhalten
  (Abschnitt 5, Punkt 7), ist aber nicht mehr die einzige Absicherung.

### Was weiterhin offen bzw. fehleranfällig bleibt

- **Die Handshake-Messung selbst ist punktuell, nicht kontinuierlich:** Sie
  läuft einmal unmittelbar vor dem Deploy (bzw. vor jedem Baseline-Lauf), nicht
  fortlaufend während der gesamten Inferenzdauer. Bei sehr langen Läufen
  (Imagenette, ~116 s) kann die Pi-Uhr zwischen Messzeitpunkt und
  Fensterende geringfügig weiterdriften; `offset_std_s` (Streuung über die
  sieben Pings) ist dafür ein Anhaltspunkt, aber keine Garantie für Drift
  *während* der Inferenz selbst. Für die hier relevanten LAN-Verhältnisse
  (Millisekunden-Größenordnung, siehe Abschnitt 3) ist das vernachlässigbar,
  bleibt aber die verbleibende methodische Näherung.
- Der `handle_time_sync`-Handler auf dem Pi muss latenzfrei antworten, um die
  Symmetrie-Annahme der NTP-Formel (PC→Pi-Latenz ≈ Pi→PC-Latenz) zu erfüllen;
  eine künftige Änderung an `deployment-receiver.py`, die vor der
  Zeitstempel-Erfassung blockierende Arbeit einführt, würde die
  Offset-Messung verzerren, ohne dass das ohne erneuten Blick in den Code
  auffällt.
- Verbesserungsvorschlag (aus der Vorfassung, jetzt umgesetzt): ein NATS-
  Zeit-Handshake mit protokolliertem Offset pro Experiment — siehe oben.
- **Neue, aber kleine Restrisiko-Fläche durch die Baseline-Blöcke:** Ihre
  „davor/danach"-Zuordnung zu einer Zelle setzt voraus, dass Baseline-Blöcke
  und echte Experimente strikt sequenziell laufen (`_run_scenario`s
  synchrone Schleife mit `await`) — aktuell durch die Pipeline-Architektur
  gegeben, aber anders als die Experiment-Zuordnung (Punkt 4) nicht durch
  einen eigenen Audit-Punkt abgesichert. Ließe sich mit geringem Aufwand in
  `verify_power_coverage.py` ergänzen: prüfe, dass jeder `baseline_NNN`-Block
  ein `recorded_at` hat, das monoton zwischen den `started_at`-Werten der
  umgebenden Experimente liegt.
- `energy_per_image_J`/`avg_power_W` (Brutto, globale Baseline) enthalten
  weiterhin das komplette Inter-Frame-Delay (200 ms/Bild) im Fenster; für
  Absolutaussagen sind jetzt `net_energy_per_image_J`/`net_avg_power_W_local`
  (lokale, warme Baseline, siehe Abschnitt 2) vorzuziehen.
- Genauigkeits- und Latenzmetriken sind von der gesamten Thematik unberührt:
  Latenz wird auf dem Pi mit einer einzigen Uhr gemessen (`latency_ms`,
  software-getimt, unabhängig von der Sampling-Hardware), Accuracy benötigt
  keine Zeitbasis.

## Referenzen (Code)

| Mechanismus | Datei | Stelle |
|---|---|---|
| PC-Timestamps der Leistungsmessung | `dsp3005_reader.py` | `log_row`, Z. 111 |
| Start/Stopp der Messung um das Deployment | `benchmark_pipeline.py` | `_subscribe_deploy_collect`, ab Z. 888 |
| ID-Tag-Prüfung vor Ergebnis-Annahme | `benchmark_pipeline.py` | `handle_msg`, ab Z. 941 |
| ID-Tag beim Senden (Pi → PC) | `client/transmitter.py` | `transmit_logs` |
| Genau-einmal-Versand pro Run (Pi) | `client/deployment-receiver.py` | `transmit_current_once` |
| Fensterbestimmung `[min, max]` | `benchmark_pipeline.py` | `_inference_window`, Z. 1127 |
| Trapezintegration | `benchmark_pipeline.py` | `_integrate`, Z. 1143 |
| Pi-Ressourcen-Logger (Start nach Setup) | Szenario-`main.py` | Start Z. 947, Stopp Z. 1112 |
| Baseline-Blöcke (warm, n je Zelle) | `benchmark_pipeline.py` | `run_baseline_block`, ab Z. 816 |
| Lokale Baseline-Zuordnung (nur PC-Zeit) | `analysis/analyze_benchmark.py` | `_local_baseline_power_w`, Z. 353; `_load_baseline_blocks`, Z. 330 |
| Audit | `analysis/verify_power_coverage.py` | gesamtes Skript |
| PC↔Pi-Zeit-Handshake (Pi-Seite, Reply) | `client/deployment-receiver.py` | `handle_time_sync`, in `start_consumer` |
| PC↔Pi-Zeit-Handshake (PC-Seite, Messung) | `benchmark_pipeline.py` | `_measure_clock_offset` |
| Offset-Korrektur des Inferenzfensters | `analysis/analyze_benchmark.py` | `_offset_corrected_power_metrics` |
| Sanity-Check des gemessenen Handshake-Offsets (Audit Punkt 7) | `analysis/verify_power_coverage.py` | `audit_experiment`, Check 7 (`--offset-warn`) |
| Implementierungsplan des Handshakes | `PLAN_ZEIT_HANDSHAKE.md` | gesamtes Dokument |
