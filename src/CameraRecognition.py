import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path

# ---- RUTAS ----
base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "sign_language_model.h5"
encoder_path = base_dir / "label_encoder.pkl"

# ---- CARGAR MODELO Y LABEL ENCODER ----
model = load_model(model_path)
with open(encoder_path, "rb") as f:
    le = pickle.load(f)

# ---- CAPTURA DE VIDEO ----
cap = cv2.VideoCapture(0)  # 0 = cámara por defecto

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar frame")
        break

    # Preprocesar frame (usar una ROI más pequeña si quieres)
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1,64,64,3)

    # Predicción
    pred_probs = model.predict(img, verbose=0)
    pred_index = np.argmax(pred_probs)
    pred_class = le.inverse_transform([pred_index])[0]

    # Mostrar resultado en la ventana
    cv2.putText(frame, f"Letra: {pred_class}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento de Lengua de Signos", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
