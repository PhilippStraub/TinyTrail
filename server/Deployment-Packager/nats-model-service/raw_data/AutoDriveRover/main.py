import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import rover

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

# === Softmax ===

# def softmax(x):
#     x = np.array(x, dtype=np.float32)  # explizit Float verwenden!
#     e_x = np.exp(x - np.max(x))        # zentrieren für Stabilität
#     return e_x / e_x.sum()

# def sigmoid(x):
#     x = np.array(x, dtype=np.float32)
#     return 1 / (1 + np.exp(-x))

# def normalize_logits_exponential(x, max_value=255.0, base=5.0):
#     """
#     Exponentielle Skalierung von Logit-Werten.
#     Höhere base → stärkerer Effekt.
#     """
#     x = np.array(x, dtype=np.float32)
#     x_scaled = x / max_value  # Werte auf [0, 1] skalieren
#     return np.clip((base ** x_scaled - 1) / (base - 1), 0.0, 1.0)

# === Rover Steuerung ===
# Servo numbers
servo_FL = 9
servo_RL = 11
servo_FR = 15
servo_RR = 13
servo_MA = 0

speed = 50

def goForward():
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)
    rover.forward(speed)

def goReverse():
    rover.setServo(servo_FL, 0)
    rover.setServo(servo_FR, 0)
    rover.setServo(servo_RL, 0)
    rover.setServo(servo_RR, 0)
    rover.reverse(speed)

def goLeft():
    rover.setServo(servo_FL, -20)
    rover.setServo(servo_FR, -20)
    rover.setServo(servo_RL, 20)
    rover.setServo(servo_RR, 20)

def goRight():
    rover.setServo(servo_FL, 20)
    rover.setServo(servo_FR, 20)
    rover.setServo(servo_RL, -20)
    rover.setServo(servo_RR, -20)

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
    rover.init(0) #Rover

    while True:
        try:
            # Bild aufnehmen und rotieren
            image_pil = picam2.capture_image("main").convert("RGB").rotate(270, expand=True)
            image_np = np.array(image_pil)

            # Klassifikation
            results = classify_image(interpreter, image_np)
            #Num Score
            top_index = int(np.argmax(results))
            score = results[top_index]
            limit = 140

            #Softmax Score
            # probabilities = normalize_logits_exponential(results)
            # top_index = int(np.argmax(probabilities))
            # score = probabilities[top_index]
            # limit = 0.7
            label = labels.get(top_index, 'Unbekannt')

            print(f'Label: {label}, Score: {score:.2f}, Distance: {rover.getDistance()}')

            # Beispielaktion
            if label == "right" and score > limit:
                print("Hindernis Rechts erkannt, steuere nach links.")
                print ('Go Left', speed)
                #goLeft()
                rover.spinLeft(speed)
            elif label == "left" and score > limit:
                print("Hindernis Links erkannt, steuere nach rechts.")
                print ('Go Right', speed)
                #goRight()
                rover.spinRight(speed)
            elif label == "front" and score > limit and rover.getDistance() < 20:
                print("Hindernis vor mir erkannt.")
                print ('Brake')
                rover.brake()
                rover.spinLeft(speed)
                # rover.setServo(servo_MA, -30)

            elif label == "negative" and score > limit:
                print("Kein Hindernis erkannt, fahre.")
                print ('Forward', speed)
                rover.forward(speed)
                
            else:
                print("Nichts erkannt.")

            time.sleep(0.5)  # 2 FPS

        except KeyboardInterrupt:
            print("\n[INFO] Beendet durch Benutzer.")
            rover.cleanup() #Rover
            break
  

if __name__ == '__main__':
    main()

