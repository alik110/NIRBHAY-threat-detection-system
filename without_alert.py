import cv2
import torch
import numpy as np
import tensorflow as tf
from collections import deque
from ultralytics import YOLO

# Load models
violence_model = tf.keras.models.load_model("violence/violence_model.keras")
weapon_model = YOLO("weapon/best.pt")

# Set device for YOLO
device = "cuda" if torch.cuda.is_available() else "cpu"
weapon_model.to(device)

# Constants
frame_height, frame_width = 96, 96
sequence_length = 16
class_labels = ["NonViolence", "Violence"]
CONF_THRESHOLD = 0.5

# Frame buffer for violence detection
frame_buffer = deque(maxlen=sequence_length)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Copy frame for processing
    processed_frame = frame.copy()

    # --- Violence Detection ---
    resized_frame = cv2.resize(processed_frame, (frame_width, frame_height))
    normalized_frame = resized_frame.astype('float32') / 255.0
    frame_buffer.append(normalized_frame)

    if len(frame_buffer) == sequence_length:
        input_sequence = np.array(frame_buffer)
        input_sequence = np.expand_dims(input_sequence, axis=0)

        prediction = violence_model.predict(input_sequence, verbose=0)
        predicted_class = np.argmax(prediction)
        violence_label = class_labels[predicted_class]

        color = (0, 255, 0) if violence_label == "NonViolence" else (0, 0, 255)
        cv2.putText(frame, f"{violence_label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # --- Weapon Detection ---
    results = weapon_model.predict(frame, conf=CONF_THRESHOLD, imgsz=480, device=device, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = map(int, box.xyxy[0].tolist() + [box.conf[0], box.cls[0]])
            label = f"{weapon_model.names[cls]}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Violence & Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
