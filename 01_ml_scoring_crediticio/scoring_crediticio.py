"""
Ejemplo de Machine Learning: scoring crediticio para microcréditos.

Caso: Financiera Confianza Perú (componente 2 de la propuesta).
Modelo: XGBoost (gradient boosting) comparado con una regresión logística.
Entrada: variables del solicitante y de su negocio.
Salida: probabilidad de mora (incumplimiento > 30 días) y banda de riesgo.

Los datos son SINTÉTICOS: se generan con reglas documentadas en generar_datos(),
porque la financiera es un caso de estudio y no hay datos reales disponibles.

Uso:
    pip install numpy pandas scikit-learn xgboost matplotlib
    python scoring_crediticio.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

SEMILLA = 42
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(SALIDA, exist_ok=True)

NUMERICAS = ["edad", "antiguedad_negocio_meses", "ventas_mensuales", "monto_solicitado", "plazo_meses",
             "cuota_sobre_ingreso", "creditos_previos", "max_dias_atraso_12m", "entidades_con_deuda",
             "dias_uso_app_90d"]
CATEGORICAS = ["rubro", "region"]


def generar_datos(n=20000, semilla=SEMILLA):
    """Genera solicitudes de microcrédito con una probabilidad de mora definida por reglas."""
    r = np.random.default_rng(semilla)
    df = pd.DataFrame({
        "edad": r.integers(20, 66, n),
        "antiguedad_negocio_meses": r.gamma(2.2, 18, n).round().clip(3, 360),
        "ventas_mensuales": r.lognormal(8.3, 0.55, n).round(-1),          # soles
        "plazo_meses": r.choice([6, 9, 12, 18, 24], n, p=[.15, .15, .35, .2, .15]),
        "creditos_previos": r.poisson(1.6, n),
        "max_dias_atraso_12m": (r.exponential(6, n) * r.binomial(1, .45, n)).round(),
        "entidades_con_deuda": r.poisson(1.1, n),
        "dias_uso_app_90d": r.integers(0, 91, n),
        "rubro": r.choice(["comercio", "servicios", "produccion", "agropecuario"], n, p=[.45, .25, .15, .15]),
        "region": r.choice(["costa", "sierra", "selva"], n, p=[.5, .35, .15]),
    })
    df["monto_solicitado"] = (df["ventas_mensuales"] * r.uniform(0.6, 2.5, n)).round(-1)
    tasa_mensual = 0.035
    cuota = df["monto_solicitado"] * tasa_mensual / (1 - (1 + tasa_mensual) ** (-df["plazo_meses"]))
    df["cuota_sobre_ingreso"] = (cuota / (df["ventas_mensuales"] * 0.30)).round(3)   # margen estimado 30 %

    # Reglas que definen el riesgo (logit): el modelo debe aprenderlas a partir de los datos
    z = (-2.0
         + 1.8 * (df["cuota_sobre_ingreso"] > 0.8)                     # cuota que supera la capacidad de pago
         + 0.9 * (df["cuota_sobre_ingreso"] - 0.6)
         + 0.045 * df["max_dias_atraso_12m"]
         + 1.1 * ((df["max_dias_atraso_12m"] > 15) & (df["entidades_con_deuda"] >= 2))   # sobreendeudamiento
         + 0.25 * df["entidades_con_deuda"]
         - 0.010 * df["antiguedad_negocio_meses"]
         - 0.30 * np.minimum(df["creditos_previos"], 4)
         - 0.015 * df["dias_uso_app_90d"]
         + 0.8 * ((df["rubro"] == "agropecuario") & (df["region"] == "sierra"))   # estacionalidad agrícola
         + 0.6 * ((df["edad"] < 25) & (df["antiguedad_negocio_meses"] < 12))
         + r.normal(0, 0.35, n))
    df["mora"] = r.binomial(1, 1 / (1 + np.exp(-z)))
    return df


def banda_riesgo(p):
    """Traduce la probabilidad de mora en la decisión del motor de reglas."""
    if p < 0.15:
        return "Bajo (aprobación automática)"
    if p < 0.40:
        return "Medio (revisión del analista)"
    return "Alto (rechazo)"


def main():
    df = generar_datos()
    df.to_csv(os.path.join(SALIDA, "solicitudes_sinteticas.csv"), index=False)
    X, y = df[NUMERICAS + CATEGORICAS], df["mora"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
    print(f"Solicitudes: {len(df)} | tasa de mora: {y.mean():.1%}")

    prep = ColumnTransformer([("num", StandardScaler(), NUMERICAS),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    logit = make_pipeline(prep, LogisticRegression(max_iter=1000))
    xgb = make_pipeline(
        ColumnTransformer([("num", "passthrough", NUMERICAS),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)]),
        XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.03, min_child_weight=10, reg_lambda=5, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="auc", random_state=SEMILLA))

    resultados = {}
    for nombre, modelo in [("Regresión logística", logit), ("XGBoost", xgb)]:
        modelo.fit(X_tr, y_tr)
        p = modelo.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p)
        resultados[nombre] = (modelo, p, auc)
        print(f"{nombre:20s} AUC = {auc:.3f} | Gini = {2 * auc - 1:.3f}")

    modelo, p, auc = resultados["XGBoost"]

    # Curva ROC
    plt.figure(figsize=(5, 4))
    for (nombre, (_, pp, a)), estilo in zip(resultados.items(), ["--", "-"]):
        fpr, tpr, _ = roc_curve(y_te, pp)
        plt.plot(fpr, tpr, estilo, color="black", label=f"{nombre} (AUC = {a:.3f})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "roc.png"), dpi=200)

    # Importancia de variables
    nombres = NUMERICAS + list(modelo[0].named_transformers_["cat"].get_feature_names_out(CATEGORICAS))
    imp = pd.Series(modelo[-1].feature_importances_, index=nombres).sort_values().tail(10)
    plt.figure(figsize=(5, 4))
    imp.plot.barh(color="0.35", edgecolor="black")
    plt.xlabel("Importancia (ganancia relativa)")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "importancia.png"), dpi=200)

    # Matriz de confusión con el umbral de rechazo (0,40)
    pred = (p >= 0.40).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"Umbral 0,40 -> VP={tp} FP={fp} FN={fn} VN={tn} | recall mora={tp / (tp + fn):.1%}")

    # Salida esperada para tres solicitudes nuevas
    todos = X_te.copy()
    todos["prob_mora"] = p.round(3)
    # una solicitud de cada banda para ilustrar la salida
    nuevos = pd.concat([todos[todos.prob_mora < 0.15].head(1),
                        todos[(todos.prob_mora >= 0.15) & (todos.prob_mora < 0.40)].head(1),
                        todos[todos.prob_mora >= 0.40].head(1)])
    nuevos["banda"] = nuevos["prob_mora"].apply(banda_riesgo)
    cols = ["ventas_mensuales", "monto_solicitado", "cuota_sobre_ingreso", "max_dias_atraso_12m", "prob_mora", "banda"]
    print(nuevos[cols].to_string())
    nuevos[cols].to_csv(os.path.join(SALIDA, "ejemplo_salida.csv"), index=False)

    bandas = pd.Series(p).apply(banda_riesgo).value_counts(normalize=True)
    print("Distribución de bandas en prueba:\n", bandas.round(3).to_string())
    with open(os.path.join(SALIDA, "metricas.txt"), "w", encoding="utf-8") as f:
        for nombre, (_, _, a) in resultados.items():
            f.write(f"{nombre}: AUC={a:.3f} Gini={2 * a - 1:.3f}\n")
        f.write(f"XGBoost umbral 0,40: VP={tp} FP={fp} FN={fn} VN={tn}\n")
        f.write(bandas.round(3).to_string())


if __name__ == "__main__":
    main()
