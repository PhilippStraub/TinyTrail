# Zeitliche Zuordnung der Energiemessung: Methodik und Validität

Dieses Dokument erklärt, wie die Inferenz-Energiemessung im Benchmark zeitlich
funktioniert: welche Uhren beteiligt sind, wie das Inferenzfenster exakt
bestimmt wird, warum der fehlende explizite Uhrenabgleich die Messungen unter
den unten genannten — prüfbaren — Bedingungen nicht maßgeblich verfälscht, und
wie wir das nachweisen.

## 1. Messaufbau: zwei Uhren, ein Messgerät ohne eigene Uhr

| Komponente | Rolle | Zeitquelle |
|---|---|---|
| Voltcraft DSP-3005 | misst Spannung/Strom der Pi-Versorgung | **keine** — liefert nur V/I per SCPI |
| Benchmark-PC | pollt das DSP-3005 mit 5 Hz, stempelt jede Zeile | PC-Systemuhr (`datetime.now()`, `dsp3005_reader.py:93`) |
| Raspberry Pi | loggt CPU/RAM mit 5 Hz während der Inferenz | Pi-Systemuhr (`time.time()`, `resource_usage_*.csv`) |

Die Leistungs-Zeitreihe (`power_*.csv`) lebt also vollständig auf der PC-Uhr,
das Inferenzfenster vollständig auf der Pi-Uhr. Ein expliziter
Synchronisationsschritt (z. B. NTP-Handshake) findet nicht statt; beide
Betriebssysteme synchronisieren ihre Uhren standardmäßig selbst per NTP
(Raspberry Pi OS: `systemd-timesyncd`, Windows: `time.windows.com`).

## 2. Exakter Ablauf der Inferenz-Energiemessung

Pro Experiment führt `benchmark_pipeline.py` folgende Sequenz aus
(`_subscribe_deploy_collect`, Zeilen 898–948):

```
PC:  PowerLogger.start()          ← Messung beginnt VOR dem Deployment
PC:  Deployment an den Pi (NATS)
Pi:  Deployment empfangen, entpacken, Python-Start
Pi:  Setup: TFLite-Modell laden (eager), PCA laden, KNN-State laden
Pi:  ResourceCSVLogger.start()    ← erste Zeile sofort, dann alle 200 ms
Pi:  Inferenz-Schleife über alle Testbilder
Pi:  ResourceCSVLogger.stop()     ← letzte Zeile ≤ 0,2 s nach Schleifenende
Pi:  Merge der Logs, Ergebnis-ZIP über NATS an den PC
PC:  Ergebnis empfangen → PowerLogger.stop()
```

Das **Inferenzfenster** ist definiert als `[t0, t1] = [min, max]` der
`timestamp_unix`-Werte des Pi-Ressourcen-Logs (`benchmark_pipeline.py:1000`).
Da der Ressourcen-Logger erst nach dem vollständigen Setup startet (Kommentar
im Szenario-Code: *„Start resource logging only when the actual processing
starts"*) und im `finally` unmittelbar nach der Schleife stoppt, umschließt
das Fenster die Inferenzphase auf ±1–2 Samples (±0,2–0,4 s) genau — beide
Kanten stammen von **derselben Uhr** wie die Inferenz selbst.

Die Energie ergibt sich als Trapezintegral der Leistungs-Samples innerhalb
des Fensters (`_integrate`, `benchmark_pipeline.py:1016`). Ausgewiesen werden
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
5. **Uhrenfreie Offset-Schätzung:** Kreuzkorrelation der Pi-CPU-Last mit der
   PC-Leistungskurve; der Korrelations-Peak liefert den Restversatz δ̂ direkt
   aus den Signalen.
6. **Sensitivität:** Verschiebung des Fensters um ±2 s und Neuberechnung der
   Energie beziffert den maximal möglichen Randeffekt pro Lauf.

Validierung des Audits an synthetischen Testdaten: Ein künstlich eingebauter
Uhrenversatz von +3 s wurde per Kreuzkorrelation auf die Sekunde genau
erkannt (r = 0,98), ein simulierter NTP-Sprung von 300 s mitten im Lauf
schlug in vier unabhängigen Checks fehl; ein sauberer Lauf passierte mit
δ̂ = 0,0 s und einer Fenster-Sensitivität von 0,5 %.

**Die Validitätsaussage ist damit bedingt und prüfbar:** *Sofern* das Audit
über alle Experimente ohne FAIL läuft, ist belegt — nicht angenommen —, dass
(a) jeder Trace sein Fenster mit Reserve überdeckt, (b) kein Uhrensprung
auftrat, (c) der Restversatz klein gegen die Fensterlänge war und (d) selbst
eine Fensterverschiebung um ±2 s die Energiewerte nur im ausgewiesenen
Sub-Prozent-Bereich ändern würde. Schlägt das Audit für einzelne Läufe fehl,
sind genau diese Läufe zu verwerfen oder zu wiederholen.

## 6. Grenzen und Ausblick

- Die Synchronisation stützt sich auf die NTP-Voreinstellungen der
  Betriebssysteme; ein expliziter Prüfschritt (`timedatectl status` auf dem
  Pi vor dem Sweep) bzw. ein NATS-Zeit-Handshake mit protokolliertem Offset
  pro Experiment wäre die saubere Erweiterung für künftige Messkampagnen.
- `energy_per_image_J` enthält das Inter-Frame-Delay (200 ms/Bild) und ist
  als Fensterdurchschnitt zu berichten; für Absolutaussagen ist
  `net_energy_per_image_J` (Idle-Baseline abgezogen) vorzuziehen.
- Genauigkeits- und Latenzmetriken sind von der Thematik gänzlich unberührt:
  Latenz wird auf dem Pi mit einer einzigen Uhr gemessen, Accuracy benötigt
  keine Zeitbasis.

## Referenzen (Code)

| Mechanismus | Datei | Stelle |
|---|---|---|
| PC-Timestamps der Leistungsmessung | `dsp3005_reader.py` | `log_row`, Z. 93 |
| Start/Stopp der Messung um das Deployment | `benchmark_pipeline.py` | Z. 898–948 |
| Fensterbestimmung `[min, max]` | `benchmark_pipeline.py` | `_inference_window`, Z. 1000 |
| Trapezintegration | `benchmark_pipeline.py` | `_integrate`, Z. 1016 |
| Pi-Ressourcen-Logger (Start nach Setup) | Szenario-`main.py` | Z. 961–968, Stopp Z. 1128 |
| Audit | `analysis/verify_power_coverage.py` | gesamtes Skript |
