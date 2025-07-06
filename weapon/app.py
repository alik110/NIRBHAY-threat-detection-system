import cv2
import torch
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 1 if external camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce if needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS to avoid unnecessary lag

CONF_THRESHOLD = 0.5  # Lowering this may help speed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO inference (with GPU if available)
    results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=480, device=device, verbose=False)  

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = map(int, box.xyxy[0].tolist() + [box.conf[0], box.cls[0]])
            label = f"{model.names[cls]}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Weapon Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
