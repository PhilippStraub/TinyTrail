import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Füge das Hauptverzeichnis zum Python-Path hinzu
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(root_dir))

import cv2  # Für das Laden und Vorverarbeiten der Bilder
import numpy as np
from typing import Optional

# Importiere die Hilfsfunktionen aus examples/utils/preprocessors.py
from examples.utils.preprocessors import resize_image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        TFLITE_RUNTIME_AVAILABLE = True
    except ImportError:
        TFLITE_RUNTIME_AVAILABLE = False

# from tinylcm_python.knn import LightweightKNN  # nur für CLASSIFIER_TYPE == "knn" benötigt (hier: "nearest_centroid")
from tinylcm_python.prototype_classifier import (
    NearestCentroidClassifier,
    # KMeansPrototypeClassifier,  # nur für CLASSIFIER_TYPE == "kmeans" benötigt (hier: "nearest_centroid")
)
from tinylcm_python.scaler_pca import StandardScalerPCATransformer
from tinylcm_python.qparams import get_qparams
# ensure_directory_exists is not needed; use pathlib directly

# --- KONFIGURATION ---
# SCENARIO 2: Train WITH lego, ball will trigger drift
# Pfade zu initialen Bilddaten
# Prefer repo-level Imagenette dataset used by training scripts; fall back to local scenario copy if present.
INITIAL_IMAGE_DATA_DIR = root_dir / "examples" / "dataset_imagenette" / "train"
if not INITIAL_IMAGE_DATA_DIR.exists():
    INITIAL_IMAGE_DATA_DIR = current_dir / "dataset_imagenette" / "train"
# Classes for Scenario 2 - INCLUDING lego (ball not in training)
CLASSES = {
    "cassette_player": INITIAL_IMAGE_DATA_DIR / "cassette_player",
    "chain_saw": INITIAL_IMAGE_DATA_DIR / "chain_saw",
    "church": INITIAL_IMAGE_DATA_DIR / "church",
    "english_springer": INITIAL_IMAGE_DATA_DIR / "english_springer",
    "french_horn": INITIAL_IMAGE_DATA_DIR / "french_horn",
    "garbage_truck": INITIAL_IMAGE_DATA_DIR / "garbage_truck",
    "gas_pump": INITIAL_IMAGE_DATA_DIR / "gas_pump",
    "golf_ball": INITIAL_IMAGE_DATA_DIR / "golf_ball",
    "parachute": INITIAL_IMAGE_DATA_DIR / "parachute",
    "tench": INITIAL_IMAGE_DATA_DIR / "tench",
}

# Pfad zum TFLite-Modell und Labels (scenario-specific)
MODEL_PATH = str(current_dir / "model/model_object.tflite")
LABELS_PATH = str(current_dir / "model/labels_object.txt")

# Pfad zum Feature-Prozessor (StandardScaler + PCA) – konsistent zur Config
FEATURE_PROCESSOR_PATH = str(current_dir / "model/feature_processor.npz")

# Konfiguration für den Feature Extractor
TARGET_IMG_SIZE = (224, 224)  # Inferenzauflösung (MobileNetV2)

# Konfiguration für den LightweightKNN
# KNN_K wird von NearestCentroidClassifier nicht verwendet: fit() setzt self.k immer
# selbst auf die Klassenanzahl (siehe prototype_classifier.py, NearestCentroidClassifier
# .fit()), unabhängig von config.json oder dieser Konstante — anders als bei KNN/Kmeans
# (siehe TESTPLAN_HARDWARE_VALIDIERUNG.md, Befund 10.1) ist das hier kein Bug, sondern
# Absicht. Deshalb bleibt der Config-Read hier auskommentiert statt aktiviert.
# with open(current_dir / "config.json") as _cfg_f:
#     KNN_K = json.load(_cfg_f)["tinylcm"]["adaptive_classifier"]["k"]
KNN_MAX_SAMPLES = 500  # Maximum samples to retain (CLI-Fallback für samples_per_class)
KNN_DISTANCE_METRIC = "euclidean"  # Metrik für den Abstandsvergleich
# KNN_USE_NUMPY = True  # (deprecated, nirgends mehr referenziert) Laufzeit nutzt immer NumPy-Vektorisierung

# Klassifikator-Typ für den initialen Zustand (muss zu config.json passen):
#   "knn"             -> LightweightKNN (alle balancierten Samples)
#   "nearest_centroid"-> NearestCentroidClassifier (1 Prototyp/Klasse = Mittelwert)
#   "kmeans"          -> KMeansPrototypeClassifier (m Prototypen/Klasse via K-Means)
CLASSIFIER_TYPE = "nearest_centroid"
# KMEANS_M_PROTOTYPES = 3  # nur für CLASSIFIER_TYPE == "kmeans" benötigt (hier: "nearest_centroid")

# Speicherort für den initialen k-NN Zustand
OUTPUT_STATE_DIR = current_dir / "initial_state"  # Scenario initial_state directory
OUTPUT_STATE_FILENAME = "knn_initial_state_objects.json"  # Zustandsdatei
OUTPUT_REFERENCE_STATS_FILENAME = "knn_reference_stats.json"  # Datei für die Referenzstatistiken
# --- ENDE KONFIGURATION ---

# Interpreter-Cache: einmal erstellen, über alle Experimente wiederverwenden, damit
# ein Benchmark-Sweep den TensorFlow-Import / die Interpreter-Allokation nur EINMAL zahlt.
_INTERPRETER = None

def get_interpreter():
    """Return a cached TFLite interpreter (created and tensors allocated once)."""
    global _INTERPRETER
    if _INTERPRETER is None:
        if TF_AVAILABLE:
            _INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH)
        elif TFLITE_RUNTIME_AVAILABLE:
            _INTERPRETER = TFLiteInterpreter(model_path=MODEL_PATH)
        else:
            raise ImportError("Weder TensorFlow noch tflite_runtime ist verfügbar. Bitte installiere eines davon.")
        _INTERPRETER.allocate_tensors()
    return _INTERPRETER

def preprocess_image_for_feature_extraction(image_path: Path, target_size: tuple) -> np.ndarray:
    """Lädt und resized ein Bild für die Feature-Extraktion (RGB uint8)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    
    # Verwende die resize_image Funktion aus utils.preprocessors
    img_resized = resize_image(img, target_size)
    
    # Konvertiere zu RGB (TFLite erwartet oft RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    return img_rgb  # Return RGB image as uint8

def extract_features_manually(
    image,
    interpreter,
    input_details,
    output_details,
    feature_transformer: Optional[StandardScalerPCATransformer] = None,
):
    """Extrahiere Features mit konsistenter Preprocessing/Quantisierung; optional PCA anwenden."""
    # 1) Preprocessing wie zur Laufzeit Low-res MobileNet: [0,1]
    img = image.astype(np.float32)
    img = (img / 127.5) - 1.0
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    # 2) Quantisierung falls INT8/UINT8
    in_dtype = input_details['dtype']
    processed_image = img
    if in_dtype in (np.int8, np.uint8):
        scale, zero_point = get_qparams(input_details)
        q = np.round(img / max(scale, 1e-12) + zero_point)
        if in_dtype == np.int8:
            processed_image = np.clip(q, -128, 127).astype(np.int8)
        else:
            processed_image = np.clip(q, 0, 255).astype(np.uint8)
    else:
        processed_image = img.astype(in_dtype)

    # 3) Inferenz
    interpreter.set_tensor(input_details['index'], processed_image)
    interpreter.invoke()

    feature_tensor = interpreter.get_tensor(output_details['index'])
    # Output-Dequantisierung wie im Runtime-Extractor
    out_dtype = output_details.get('dtype', feature_tensor.dtype)
    if out_dtype in (np.int8, np.uint8):
        o_scale, o_zero = get_qparams(output_details)
        feature_tensor = (feature_tensor.astype(np.float32) - o_zero) * o_scale
    if feature_tensor.ndim > 1 and feature_tensor.shape[0] == 1:
        feature_tensor = feature_tensor[0]

    # 4) Optional: StandardScaler + PCA anwenden
    if feature_transformer is not None:
        try:
            feats = feature_tensor.reshape(1, -1)
            transformed = feature_transformer.transform(feats)[0]
            return transformed
        except Exception as e:
            print(f"Fehler bei der Anwendung des Feature-Transformers: {e}")
            print("Verwende die ursprünglichen Features")
            return feature_tensor

    return feature_tensor

# Unbenutzt: nirgends aufgerufen, nur calculate_knn_distance_statistics_from_knn()
# (unten) wird tatsächlich verwendet. Auskommentiert statt gelöscht, da identischer
# toter Code auch in den anderen 8 create_initial_state.py-Kopien steckt.
# def calculate_knn_distance_statistics(knn_instance, all_features, all_labels):
#     """Berechne LOO-Statistiken über KNN-Distanzen (kompatibel zu Runtime-Distanzen).
#
#     - Entfernt Selbst-Nachbar (Distanz ~ 0) via Leave-One-Out.
#     - Nutzt dieselbe Distanzskala wie zur Laufzeit (euclidische Distanzen, nicht quadriert).
#     """
#     print("Berechne KNN-Distanz-Statistiken (Leave-One-Out) für initiale Referenz...")
#
#     # Sicherheitschecks
#     if knn_instance.X_train is None or len(knn_instance.X_train) == 0:
#         print("FEHLER: KNN ist leer, kann keine Distanzen berechnen")
#         return {}
#
#     k = int(getattr(knn_instance, 'k', 5))
#     metric = str(getattr(knn_instance, 'distance_metric', 'euclidean')).lower()
#     is_euclidean = 'euclidean' in metric
#
#     all_avg_distances = []
#     avg_distances_by_class = {}
#
#     print(f"Berechne Distanzen für {len(all_features)} Samples (k={k}, metric={metric})...")
#
#     X = knn_instance.X_train.astype(np.float32)
#     eps = 1e-12
#
#     for i, (feature, label) in enumerate(zip(all_features, all_labels)):
#         q = np.asarray(feature, dtype=np.float32)
#         # Vollständige Distanzberechnung zu allen Train-Samples
#         d = knn_instance._compute_distances(q)
#         d = np.asarray(d)
#
#         # Leave-One-Out: Selbsttreffer (Distanz ~ 0) entfernen
#         if is_euclidean:
#             mask = d > eps  # d sind quadrierte Distanzen
#             d_filtered = d[mask]
#         else:
#             # Cosine-Distanz (1 - cos): 0 bei identischem Vektor
#             mask = d > eps
#             d_filtered = d[mask]
#
#         if d_filtered.size == 0:
#             continue
#
#         # Wähle k nächstgelegene Distanzen
#         kk = min(k, d_filtered.shape[0])
#         idx = np.argpartition(d_filtered, kk - 1)[:kk]
#         dk = d_filtered[idx]
#
#         # Skala an Runtime anpassen: euclidische Distanz (nicht quadriert)
#         if is_euclidean:
#             dk = np.sqrt(np.maximum(dk, 0.0))
#
#         avg_distance = float(np.mean(dk))
#         all_avg_distances.append(avg_distance)
#
#         if label not in avg_distances_by_class:
#             avg_distances_by_class[label] = []
#         avg_distances_by_class[label].append(avg_distance)
#
#         if i % 50 == 0:
#             print(f"  Fortschritt: {i}/{len(all_features)} | Beispiel Label={label}, Distanzen={np.sort(dk).tolist()}, Mittel={avg_distance:.4f}")
#
#     if len(all_avg_distances) == 0:
#         print("WARNUNG: Keine Distanzen berechnet!")
#         return {}
#
#     mean_distance = float(np.mean(all_avg_distances))
#     std_distance = float(np.std(all_avg_distances))
#     min_distance = float(np.min(all_avg_distances))
#     max_distance = float(np.max(all_avg_distances))
#
#     print("Globale KNN-Distanz-Statistiken (LOO, Runtime-Skala):")
#     print(f"  Mean: {mean_distance:.4f}")
#     print(f"  Std: {std_distance:.4f}")
#     print(f"  Min: {min_distance:.4f}")
#     print(f"  Max: {max_distance:.4f}")
#
#     class_stats = {}
#     for label, distances in avg_distances_by_class.items():
#         class_mean = float(np.mean(distances))
#         class_std = float(np.std(distances))
#         class_stats[label] = {
#             'mean': class_mean,
#             'std': class_std,
#             'min': float(np.min(distances)),
#             'max': float(np.max(distances)),
#             'count': len(distances)
#         }
#         print(f"Klasse {label}: Mean={class_mean:.4f}, Std={class_std:.4f}, Count={len(distances)}")
#
#     return {
#         'reference_mean': mean_distance,
#         'reference_std': std_distance,
#         'min_distance': min_distance,
#         'max_distance': max_distance,
#         'sample_count': len(all_avg_distances),
#         'class_statistics': class_stats,
#         'generation_timestamp': datetime.now().isoformat()
#     }


def calculate_knn_distance_statistics_from_knn(knn_instance):
    """Berechne Referenzstatistiken exakt über die im KNN gespeicherten Samples (LOO).

    - Nutzt Leave-One-Out ohne eps-Heuristik (eigener Index wird ausgeschlossen).
    - Skaliert Euclidische Distanzen konsistent zur Laufzeit (sqrt der quadrierten Distanzen).
    """
    if knn_instance.X_train is None or knn_instance.X_train.shape[0] == 0:
        print("FEHLER: KNN ist leer, kann keine Distanzen berechnen")
        return {}

    X = knn_instance.X_train.astype(np.float32)
    y = knn_instance.y_train
    n = X.shape[0]
    k = min(int(getattr(knn_instance, 'k', 5)), max(1, n - 1))
    metric = str(getattr(knn_instance, 'distance_metric', 'euclidean')).lower()
    is_euclidean = 'euclidean' in metric

    all_avg = []
    per_class = {}

    for i in range(n):
        d = knn_instance._compute_distances(X[i])  # squared euclidean for EUCLIDEAN path
        d = np.asarray(d)
        # Leave-One-Out: eigenen Index ausschließen
        d[i] = np.inf
        # k kleinste
        idx = np.argpartition(d, k)[:k]
        dk = d[idx]
        if is_euclidean:
            dk = np.sqrt(np.maximum(dk, 0.0))
        avg = float(np.mean(dk))
        all_avg.append(avg)
        c = y[i]
        per_class.setdefault(c, []).append(avg)

    ref_mean = float(np.mean(all_avg))
    ref_std = float(np.std(all_avg))
    cls_stats = {
        c: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "count": len(v),
        }
        for c, v in per_class.items()
    }

    return {
        "reference_mean": ref_mean,
        "reference_std": ref_std,
        "min_distance": float(np.min(all_avg)),
        "max_distance": float(np.max(all_avg)),
        "sample_count": len(all_avg),
        "class_statistics": cls_stats,
        "generation_timestamp": datetime.now().isoformat(),
    }

def build_initial_state(samples_per_class=None, seed=42, no_pca=False,
                        feature_processor_path=None, output_dir=None,
                        interpreter=None):
    """Build and save the initial KNN state for one experiment.

    Importable and reusable in-process: a benchmark sweep can call this repeatedly
    while paying the TensorFlow import only once (the interpreter is cached via
    get_interpreter()). The CLI entry point main() preserves standalone behavior.
    """

    # Resolve paths from args (fall back to module-level defaults)
    output_state_dir = output_dir if output_dir else OUTPUT_STATE_DIR
    feature_processor_path = feature_processor_path if feature_processor_path else FEATURE_PROCESSOR_PATH

    print("=" * 50)
    print("Creating initial KNN state for Imagenette")
    print("Classes:", list(CLASSES.keys()))
    if samples_per_class:
        print(f"Samples/class: {samples_per_class}")
    print(f"PCA: {'disabled' if no_pca else 'enabled'}")
    print("=" * 50)
    # Reproducible reference sample selection
    np.random.seed(seed)
    print(f"Random seed: {seed}")
    # Ensure output directory exists
    output_state_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialisiere TFLite Interpreter
    print(f"Initialisiere TFLite Interpreter mit Modell: {MODEL_PATH}")
    
    # Lade das Model mit dem korrekten Interpreter
    if interpreter is None:
        interpreter = get_interpreter()

    interpreter.allocate_tensors()
    
    # Hole Input- und Output-Details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"Input Shape: {input_details['shape']}")
    print(f"Output Shape: {output_details['shape']}")
    print(f"Input Dtype: {input_details['dtype']}")
    
    # WICHTIG: Beachte die Quantisierungsparameter, wenn vorhanden
    if 'quantization' in input_details and len(input_details['quantization']) == 4:
        scale, zero_point = input_details['quantization'][0], input_details['quantization'][1]
        print(f"Input Quantisierung: scale={scale}, zero_point={zero_point}")
    elif 'quantization' in input_details and len(input_details['quantization']) == 2:
        scale, zero_point = input_details['quantization']
        print(f"Input Quantisierung: scale={scale}, zero_point={zero_point}")

    # Erstelle einen Dummy-Input und verarbeite ihn konsistent: [-1,1] + Quantisierung
    dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy = dummy_rgb.astype(np.float32)
    dummy = (dummy / 127.5) - 1.0
    if input_details['dtype'] in (np.int8, np.uint8):
        scale, zero_point = 1.0, 0
        if 'quantization' in input_details and input_details['quantization'] is not None:
            q = input_details['quantization']
            if isinstance(q, (list, tuple)) and len(q) >= 2:
                scale, zero_point = float(q[0] or 1.0), int(q[1] or 0)
        if 'quantization_parameters' in input_details:
            qp = input_details['quantization_parameters']
            if qp and 'scales' in qp and len(qp['scales']) > 0:
                scale = float(qp['scales'][0]) or scale
            if qp and 'zero_points' in qp and len(qp['zero_points']) > 0:
                zero_point = int(qp['zero_points'][0])
        q = np.round(dummy / scale + zero_point)
        if input_details['dtype'] == np.int8:
            dummy = np.clip(q, -128, 127).astype(np.int8)
        else:
            dummy = np.clip(q, 0, 255).astype(np.uint8)
    else:
        dummy = dummy.astype(input_details['dtype'])
    dummy = np.expand_dims(dummy, axis=0)

    # Setze den Input-Tensor mit dem korrekt verarbeiteten Dummy-Input
    interpreter.set_tensor(input_details['index'], dummy)
    
    # Führe die Inferenz durch
    interpreter.invoke()
    
    # Extrahiere Features vom Dummy-Input
    dummy_output = interpreter.get_tensor(output_details['index'])
    
    # Bestimme die Feature-Dimension (entferne Batch-Dimension wenn vorhanden)
    if len(dummy_output.shape) > 1:
        feature_dimension = dummy_output.shape[1]
    else:
        feature_dimension = dummy_output.shape[0]
        
    print(f"Ermittelte Feature-Dimension: {feature_dimension}")

    # 2. Extrahiere Features für deine initialen Bilder
    all_features = []
    all_labels = []
    all_timestamps = []

    print("Extrahiere Features aus initialen Bildern...")

    # Lade Feature-Transformer (StandardScaler + PCA), falls vorhanden und nicht deaktiviert
    feature_transformer = None
    if not no_pca:
        try:
            if os.path.exists(feature_processor_path):
                print(f"Lade Feature-Transformer: {feature_processor_path}")
                feature_transformer = StandardScalerPCATransformer(
                    processor_path=feature_processor_path,
                    strict=True
                )
            else:
                print(f"WARNUNG: Feature-Transformer nicht gefunden: {feature_processor_path}")
        except Exception as e:
            print(f"WARNUNG: Konnte Feature-Transformer nicht laden: {e}")
            feature_transformer = None
    else:
        print("PCA deaktiviert (--no-pca). Verwende rohe TFLite-Features.")
    for ci, (label_name, image_folder_path) in enumerate(CLASSES.items()):
        if not image_folder_path.is_dir():
            print(f"WARNUNG: Ordner für Klasse '{label_name}' nicht gefunden: {image_folder_path}")
            continue
        
        image_count = 0
        # Unterstütze mehrere Bildformate. Sortiert für deterministische Auswahl.
        # Imagenette ships uppercase .JPEG; collect common extensions and de-dup
        # (case-insensitive filesystems can match a pattern more than once).
        _seen = set()
        image_files = []
        for _pat in ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.JPG", "*.PNG"):
            for _f in sorted(image_folder_path.glob(_pat)):
                if _f not in _seen:
                    _seen.add(_f)
                    image_files.append(_f)

        # Nur die tatsächlich benötigten Samples extrahieren. Der train-Ordner hat
        # 5000 Bilder/Klasse; ohne diese Vorauswahl würden pro Experiment alle 50k
        # Features extrahiert und danach auf samples_per_class verworfen
        # (50–2500x Mehraufwand). Pro Klasse seeded → über seed reproduzierbar.
        if samples_per_class and len(image_files) > samples_per_class:
            rng = np.random.default_rng(seed + ci)
            keep = rng.choice(len(image_files), size=samples_per_class, replace=False)
            image_files = [image_files[i] for i in sorted(keep)]
        
        if not image_files:
            print(f"WARNUNG: Keine Bilder im Ordner '{label_name}' gefunden: {image_folder_path}")
            continue
            
        for image_file in image_files:
            try:
                print(f"  Verarbeite: {image_file} für Label: {label_name}")
                
                # Preprocessing - Lade und resize das Bild
                preprocessed_img = preprocess_image_for_feature_extraction(image_file, TARGET_IMG_SIZE)
                
                # Extrahiere Features mit konsistentem Preprocessing + optionaler PCA-Transformation
                features = extract_features_manually(
                    preprocessed_img,
                    interpreter,
                    input_details,
                    output_details,
                    feature_transformer=feature_transformer
                )
                
                # Füge Daten hinzu
                all_features.append(features.flatten())  # Flatten, um konsistente 1D-Features zu haben
                all_labels.append(label_name)
                all_timestamps.append(time.time())
                image_count += 1
            except Exception as e:
                print(f"    Fehler bei der Verarbeitung von {image_file}: {e}")
                import traceback
                traceback.print_exc()
    
    if not all_features:
        print("FEHLER: Keine Features extrahiert. Bitte überprüfe Pfade und Bilder.")
        return

    # Konvertiere Feature-Liste in ein NumPy-Array
    initial_features_np = np.vstack(all_features)
    print(f"Form der verarbeiteten Features: {initial_features_np.shape}")

    # 3. Initialisiere und trainiere den LightweightKNN
    #
    # IMPORTANT: max_samples must hold ALL reference samples we are about to add
    # (samples_per_class × num_classes). The KNN evicts oldest-first (FIFO by
    # timestamp) once it exceeds max_samples — with a too-small cap, the classes
    # added first are silently dropped, leaving the model unable to predict them.
    n_classes = len(CLASSES)
    samples_per_class = samples_per_class if samples_per_class else KNN_MAX_SAMPLES // n_classes
    knn_max_samples = samples_per_class * n_classes
    if CLASSIFIER_TYPE == "nearest_centroid":
        knn = NearestCentroidClassifier(
            max_samples=knn_max_samples,
            distance_metric=KNN_DISTANCE_METRIC,
        )
        print("Klassifikator: NearestCentroidClassifier (1 Prototyp pro Klasse)")
    # CLASSIFIER_TYPE ist in dieser Datei fest auf "nearest_centroid" gesetzt
    # (Zeile ~87) — die anderen beiden Zweige des gemeinsamen Templates sind hier
    # toter Code und nur zur Portierung in die anderen Kopien auskommentiert.
    # elif CLASSIFIER_TYPE == "kmeans":
    #     knn = KMeansPrototypeClassifier(
    #         m_prototypes=KMEANS_M_PROTOTYPES,
    #         k=KNN_K,
    #         max_samples=knn_max_samples,
    #         distance_metric=KNN_DISTANCE_METRIC,
    #     )
    #     print(f"Klassifikator: KMeansPrototypeClassifier ({KMEANS_M_PROTOTYPES} Prototypen pro Klasse)")
    # else:
    #     knn = LightweightKNN(
    #         k=KNN_K,
    #         max_samples=knn_max_samples,
    #         distance_metric=KNN_DISTANCE_METRIC
    #     )
    #     print("Klassifikator: LightweightKNN (alle balancierten Samples)")

    # Balanciere die Klassen, damit alle gleichmäßig vertreten sind
    # Gruppiere Samples nach Klassen
    features_by_class = {}
    labels_by_class = {}
    timestamps_by_class = {}

    for i, label in enumerate(all_labels):
        if label not in features_by_class:
            features_by_class[label] = []
            labels_by_class[label] = []
            timestamps_by_class[label] = []

        features_by_class[label].append(initial_features_np[i])
        labels_by_class[label].append(label)
        timestamps_by_class[label].append(all_timestamps[i])

    # Finde die kleinste Klasse
    min_class_size = min(len(samples) for samples in features_by_class.values())
    print(f"Balanciere Klassen auf {min_class_size} Samples pro Klasse")

    # Erstelle balancierte Arrays
    balanced_features = []
    balanced_labels = []
    balanced_timestamps = []

    for label, features in features_by_class.items():
        # Wähle zufällig Samples aus jeder Klasse, maximal min_class_size
        indices = np.random.choice(len(features), min(len(features), min_class_size), replace=False)

        for i in indices:
            balanced_features.append(features_by_class[label][i])
            balanced_labels.append(labels_by_class[label][i])
            balanced_timestamps.append(timestamps_by_class[label][i])

    # Konvertiere zurück zu NumPy-Array
    balanced_features_np = np.array(balanced_features)

    # Statt knn.fit() verwenden wir hier eine kontrollierte Befüllung
    # um sicherzustellen, dass jede Klasse gleichmäßig vertreten ist
    print(f"Befülle KNN kontrolliert mit einer festen Anzahl von Samples pro Klasse")
    
    # samples_per_class + knn_max_samples were computed above (before KNN creation)
    print(f"Verwende {samples_per_class} Samples pro Klasse für insgesamt {knn_max_samples} Samples")
    
    # Reorganisiere balanced_features_np nach Klassen
    features_by_class = {}
    for i, label in enumerate(balanced_labels):
        if label not in features_by_class:
            features_by_class[label] = {
                'features': [],
                'timestamps': []
            }
        features_by_class[label]['features'].append(balanced_features_np[i])
        features_by_class[label]['timestamps'].append(balanced_timestamps[i])
    
    # Füge Samples pro Klasse hinzu und sammle Auswahl-Indizes für Reproduzierbarkeit
    selected_indices_per_class = {}
    per_class_counts = {}
    selected_features = []  # für Prototyp-Klassifikatoren (fit über alle Samples)
    selected_labels = []
    for label, data in features_by_class.items():
        # Begrenze auf samples_per_class
        count = min(len(data['features']), samples_per_class)
        print(f"Füge {count} Samples für Klasse '{label}' hinzu")

        # Wähle zufällige count Samples für jede Klasse (besseres Balancing)
        idx = np.random.permutation(len(data['features']))[:count]
        for i in idx:
            selected_features.append(data['features'][i])
            selected_labels.append(label)
            # Nur für CLASSIFIER_TYPE == "knn" relevant (hier: "nearest_centroid", nie wahr):
            # if CLASSIFIER_TYPE == "knn":
            #     knn.add_sample(
            #         feature=data['features'][i],
            #         label=label,
            #         timestamp=time.time()
            #     )
        selected_indices_per_class[label] = idx.tolist()
        per_class_counts[label] = count

    # Prototyp-Klassifikatoren berechnen ihre Prototypen (Centroids / K-Means-Center)
    # aus allen ausgewählten Samples via fit() und speichern nur diese Prototypen.
    if CLASSIFIER_TYPE in ("nearest_centroid", "kmeans"):
        knn.fit(np.asarray(selected_features, dtype=np.float32), selected_labels)
    
    # Hole die Size und Klassen vom KNN
    knn_state = knn.get_state()
    training_size = len(knn_state['X_train'])
    classes = list(set(all_labels))
    
    print(f"Initialer k-NN trainiert mit {training_size} Samples für Klassen: {classes}.")

    # 4. Speichere den Zustand des k-NN
    knn_state_dict = knn.get_state()
    
    # Format für den StateManager
    # Wir müssen sicherstellen, dass wir die tatsächliche Feature-Dimension nach PCA-Transformation verwenden
    # Die samples im KNN (X_train) haben die richtige Dimension
    actual_feature_dimension = len(knn_state_dict['X_train'][0]) if knn_state_dict['X_train'] else feature_dimension
    
    # Prüfen, ob die Dimension der transformierten Features korrekt ist
    print(f"Originale Feature-Dimension vom TFLite-Modell: {feature_dimension}")
    print(f"Tatsächliche Feature-Dimension nach Feature-Prozessor: {actual_feature_dimension}")
    
    state_to_save = {
        "classifier": knn_state_dict,
        "metadata": {
            "description": f"Initial KNN state for Imagenette with classes: {list(CLASSES.keys())}",
            "creation_date_iso": datetime.now().isoformat(),
            "feature_dimension": actual_feature_dimension,  # Verwende die tatsächliche Dimension nach Transformation
            "original_dimension": feature_dimension,  # Speichere auch die ursprüngliche Dimension
            "feature_processor": "StandardScaler + PCA(256)" if os.path.exists(FEATURE_PROCESSOR_PATH) else "None",
            "source_model": MODEL_PATH,
            "sampling_seed": seed,
            "per_class_counts": per_class_counts,
            "selected_indices_per_class": selected_indices_per_class
        }
    }
    
    output_file_path = output_state_dir / OUTPUT_STATE_FILENAME
    try:
        with open(output_file_path, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        print(f"Initialer k-NN Zustand erfolgreich gespeichert in: {output_file_path}")
    except Exception as e:
        print(f"FEHLER beim Speichern des k-NN Zustands: {e}")
    
    # 5. Berechne und speichere Referenzstatistiken für KNNDistanceMonitor
    print("\nBerechne Referenzstatistiken für KNNDistanceMonitor...")
    
    # Berechne Referenzstatistiken exakt auf Basis der im KNN gespeicherten Samples (LOO)
    reference_stats = calculate_knn_distance_statistics_from_knn(knn)
    
    if reference_stats:
        # Füge Metadaten hinzu
        reference_stats.update({
            "description": f"Reference statistics for KNNDistanceMonitor based on {len(balanced_features_np)} balanced samples",
            "model_path": MODEL_PATH,
            "knn_k": knn.k,  # tatsächlicher Laufzeit-k (Klassenanzahl, nicht die tote KNN_K-Konstante)
            "feature_dimension": actual_feature_dimension,
            "classes": list(set(balanced_labels))
        })
        
        # Speichere die Referenzstatistiken
        reference_stats_path = output_state_dir / OUTPUT_REFERENCE_STATS_FILENAME
        try:
            with open(reference_stats_path, 'w') as f:
                json.dump(reference_stats, f, indent=4)
            print(f"Referenzstatistiken erfolgreich gespeichert in: {reference_stats_path}")
        except Exception as e:
            print(f"FEHLER beim Speichern der Referenzstatistiken: {e}")
    else:
        print("Keine Referenzstatistiken berechnet oder gespeichert")

def main():
    parser = argparse.ArgumentParser(description="Create initial KNN state for benchmark")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Number of reference samples per class (default: max_samples/num_classes)")
    parser.add_argument("--no-pca", action="store_true",
                        help="Disable PCA feature transformation (raw TFLite features)")
    parser.add_argument("--feature-processor-path", type=str, default=None,
                        help="Path to a specific feature_processor.npz (overrides default)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save initial state files (default: ./initial_state)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible reference sample selection")
    args = parser.parse_args()
    build_initial_state(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        no_pca=args.no_pca,
        feature_processor_path=args.feature_processor_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
