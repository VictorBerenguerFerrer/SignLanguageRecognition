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

# ---- FUNCIÓN PARA PREDECIR UNA IMAGEN ----
def predict_image(image_path: Path):
    print(f"Procesando: {image_path}")
    
    # Leer imagen con OpenCV
    img = cv2.imread(str(image_path.resolve()))
    if img is None:
        print("No se pudo leer la imagen:", image_path)
        return None
    
    # Preprocesar igual que en el entrenamiento
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1,64,64,3)
    
    # Predecir
    pred_probs = model.predict(img)
    pred_index = np.argmax(pred_probs)
    pred_class = le.inverse_transform([pred_index])[0]
    
    print("Predicción:", pred_class)
    print("Probabilidades:", pred_probs[0])
    return pred_class

# ---- EJEMPLO DE USO ----
if __name__ == "__main__":
    test_image = base_dir / "SignLanguageDataset" / "dataset_split" / "images" / "test" / "IMG_7401.JPG"
    prediction = predict_image(test_image)
