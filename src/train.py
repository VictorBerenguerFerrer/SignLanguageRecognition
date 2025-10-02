# train.py
import numpy as np
import cv2
import pickle
from pathlib import Path
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- RUTAS ----
base_dir = Path(__file__).resolve().parent.parent  # retrocede desde src/
images_dir = base_dir / "SignLanguageDataset" / "dataset_split" / "images"
labels_dir = base_dir / "SignLanguageDataset" / "dataset_split" / "labels"
model_save_path = base_dir / "sign_language_model.h5"

# ---- CARGAR TODOS LOS LABELS EN UN DICCIONARIO ----
labels_dict = {}
for lbl_file in labels_dir.rglob("*.txt"):  # recorre todas las subcarpetas
    labels_dict[lbl_file.stem] = lbl_file

# ---- CARGAR IMÁGENES Y SUS LABELS ----
X = []
y = []

for img_file in images_dir.rglob("*"):  # recorre subcarpetas train/test/val
    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        # Leer imagen
        img = cv2.imread(str(img_file))
        if img is None:
            print("No se pudo leer:", img_file)
            continue
        img = cv2.resize(img, (64, 64))
        X.append(img)

        # Buscar label
        label_file = labels_dict.get(img_file.stem)
        if label_file:
            with open(label_file, "r") as f:
                label = f.read().strip()
            y.append(label)
        else:
            print("No se encontró label para:", img_file)

# ---- VERIFICAR CANTIDADES ----
print("Número de imágenes cargadas:", len(X))
print("Número de labels cargadas:", len(y))

if len(X) == 0 or len(y) == 0:
    raise ValueError("No se cargaron imágenes o labels. Revisa rutas y nombres de archivos.")

# ---- PREPROCESAMIENTO ----
X = np.array(X) / 255.0
y = np.array(y)

# ---- ENCODER PARA LABELS ----
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---- SPLIT EN TRAIN Y TEST ----
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ---- DEFINIR MODELO CNN ----
num_classes = len(le.classes_)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---- ENTRENAR ----
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=32)

# ---- GUARDAR MODELO ----
model.save(model_save_path)
print(f"Modelo guardado en: {model_save_path}")


# Guardar LabelEncoder para usar en predict.py
encoder_path = base_dir / "label_encoder.pkl"
with open(encoder_path, "wb") as f:
    pickle.dump(le, f)

print(f"LabelEncoder guardado en: {encoder_path}")
