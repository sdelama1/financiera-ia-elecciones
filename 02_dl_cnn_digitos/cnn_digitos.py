"""
Ejemplo de Deep Learning: red neuronal convolucional (CNN) para reconocer dígitos manuscritos.

Datos: conjunto "Optical Recognition of Handwritten Digits" (UCI), incluido en scikit-learn
(1797 imágenes de 8 x 8 píxeles, 10 clases). Se eligió porque viene con la librería y no
requiere descargas.

Uso:
    pip install numpy scikit-learn keras "jax[cpu]" matplotlib
    KERAS_BACKEND=jax python cnn_digitos.py      (en Windows: set KERAS_BACKEND=jax)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEMILLA = 42
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(SALIDA, exist_ok=True)
keras.utils.set_random_seed(SEMILLA)


def construir_modelo():
    """CNN: dos bloques convolucionales y una capa densa de clasificación."""
    return keras.Sequential([
        keras.Input(shape=(8, 8, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])


def main():
    d = load_digits()
    X = (d.images / 16.0).astype("float32")[..., None]       # normaliza a [0, 1]
    y = d.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)

    modelo = construir_modelo()
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    modelo.summary()
    hist = modelo.fit(X_tr, y_tr, validation_split=0.15, epochs=80, batch_size=32, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

    perdida, exactitud = modelo.evaluate(X_te, y_te, verbose=0)
    pred = modelo.predict(X_te, verbose=0).argmax(1)
    print(f"Épocas entrenadas: {len(hist.history['loss'])}")
    print(f"Exactitud en prueba: {exactitud:.4f} | pérdida: {perdida:.4f}")
    print(classification_report(y_te, pred, digits=3))

    # Curvas de aprendizaje
    plt.figure(figsize=(5, 3.6))
    plt.plot(hist.history["accuracy"], "-", color="black", label="Entrenamiento")
    plt.plot(hist.history["val_accuracy"], "--", color="black", label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Exactitud")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "curvas_aprendizaje.png"), dpi=200)

    # Matriz de confusión
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(4.4, 4))
    plt.imshow(cm, cmap="Greys")
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Dígito predicho")
    plt.ylabel("Dígito real")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "matriz_confusion.png"), dpi=200)

    # Ejemplos de predicción
    fig, ax = plt.subplots(1, 8, figsize=(8, 1.4))
    for k in range(8):
        ax[k].imshow(X_te[k, ..., 0], cmap="Greys")
        ax[k].set_title(f"real {y_te[k]}\npred {pred[k]}", fontsize=7)
        ax[k].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "ejemplos.png"), dpi=200)

    with open(os.path.join(SALIDA, "metricas.txt"), "w", encoding="utf-8") as f:
        f.write(f"Parámetros: {modelo.count_params()}\nÉpocas: {len(hist.history['loss'])}\n")
        f.write(f"Exactitud prueba: {exactitud:.4f}\nErrores: {int((pred != y_te).sum())} de {len(y_te)}\n")
    modelo.save(os.path.join(SALIDA, "cnn_digitos.keras"))


if __name__ == "__main__":
    main()
