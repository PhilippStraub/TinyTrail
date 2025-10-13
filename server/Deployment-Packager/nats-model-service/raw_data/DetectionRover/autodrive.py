import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# === Modellpfad ===
modelPath = os.path.expanduser('~/AutoDriveRover/model.tflite')

# === Labels ===
labels = {
    0: 'right',
    1: 'negative',
    2: 'left',
    3: 'front'
}

# === TFLite Interpreter laden ===
def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# === Bild klassifizieren ===
def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Vorverarbeitung
    resized = cv2.resize(image, (width, height))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)  # ggf. float32

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

# === Hauptfunktion ===
def main():
    # Kamera konfigurieren
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 960)},
        controls={"FrameDurationLimits": (500000, 500000)}  # 2 FPS
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # kurze Startverzögerung

    # Modell laden
    interpreter = load_interpreter(modelPath)

    print("[INFO] Starte Klassifikation...")

    while True:
        try:
            # Bild aufnehmen und rotieren
            image_pil = picam2.capture_image("main").convert("RGB").rotate(270, expand=True)
            image_np = np.array(image_pil)

            # Klassifikation
            results = classify_image(interpreter, image_np)
            top_index = int(np.argmax(results))
            score = results[top_index]
            label = labels.get(top_index, 'Unbekannt')

            print(f'Label: {label}, Score: {score:.2f}')

            # Beispielaktion
            if label == "right" and score > 0.7:
                print("Hindernis Rechts erkannt, steuere nach links.")
            elif label == "left" and score > 0.7:
                print("Hindernis Links erkannt, steuere nach rechts.")
            elif label == "front" and score > 0.7:
                print("Hindernis vor mir erkannt.")
            elif label == "negative" and score > 0.7:
                print("Kein Hindernis erkannt, fahre.")
            else:
                print("Nichts erkannt.")

            time.sleep(0.5)  # 2 FPS

        except KeyboardInterrupt:
            print("\n[INFO] Beendet durch Benutzer.")
            break

if __name__ == '__main__':
    main()
