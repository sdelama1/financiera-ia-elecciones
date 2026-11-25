"""
Demostración del sistema de predicción en tiempo real para la segunda vuelta presidencial
del Perú (7 de junio de 2026).

Modos:
    python demo_tiempo_real.py               -> flujo SIMULADO (no requiere credenciales)
    python demo_tiempo_real.py --x           -> flujo real desde la API de X (requiere X_BEARER_TOKEN)

En modo simulado el apoyo latente lo fija el simulador (empate de 50 % con oscilación); la gráfica
compara ese valor conocido con la estimación del predictor. Sirve para validar el algoritmo, no
para afirmar quién ganó la elección.
"""
import os
import sys
import time
from itertools import groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fuentes import FlujoX, SimuladorRedes
from predictor import PredictorTiempoReal
from sentimiento import AnalizadorSentimiento

CANDIDATOS = [
    {"id": "fujimori", "nombre": "Keiko Fujimori (Fuerza Popular)",
     "terminos": ["Keiko", "Fujimori", "@KeikoFujimori", "#Keiko2026", "Fuerza Popular"]},
    {"id": "sanchez", "nombre": "Roberto Sánchez (Juntos por el Perú)",
     "terminos": ["Roberto Sánchez", "Sánchez", "#Sanchez2026", "Juntos por el Perú", "JPP"]},
]
APOYO_BASE, AMPLITUD = 0.50, 0.03          # escenario simulado de empate técnico (primer candidato)
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(SALIDA, exist_ok=True)


def main():
    real = "--x" in sys.argv
    analizador = AnalizadorSentimiento(usar_transformer=True)
    print("Clasificador de sentimiento:", analizador.nombre)
    predictor = PredictorTiempoReal(CANDIDATOS, analizador, prior=(0.5, 0.5), fuerza_prior=200, olvido=0.85)

    if real:
        flujo = FlujoX(CANDIDATOS)
        flujo.configurar_reglas()
    else:
        flujo = SimuladorRedes(CANDIDATOS, APOYO_BASE, AMPLITUD, horas=24, pubs_por_hora=400)

    A = CANDIDATOS[0]["id"]
    for hora, grupo in groupby(flujo, key=lambda p: p.fecha.strftime("%d/%m %H:00")):
        f = predictor.actualizar(list(grupo), hora)
        print(f"[{hora}] pubs={f['publicaciones']:4d}  {A}={f[f'estimacion_{A}']:.1%} "
              f"(IC90 {f['ic90_inf']:.1%}-{f['ic90_sup']:.1%})  P(gana {A})={f[f'prob_gana_{A}']:.2f}")
        if real:
            time.sleep(0)

    df = pd.DataFrame(predictor.historial)
    if not real:
        df["apoyo_latente_simulado"] = [flujo.apoyo_latente(h) for h in range(len(df))]
        mae = (df[f"estimacion_{A}"] - df["apoyo_latente_simulado"]).abs().mean()
        print(f"Error absoluto medio frente al apoyo latente simulado: {mae * 100:.2f} puntos")
    df.to_csv(os.path.join(SALIDA, "estimaciones.csv"), index=False)

    plt.figure(figsize=(7, 3.6))
    x = range(len(df))
    plt.fill_between(x, df["ic90_inf"] * 100, df["ic90_sup"] * 100, color="0.85", label="IC 90 %")
    plt.plot(x, df[f"estimacion_{A}"] * 100, "-", color="black", label="Estimación del predictor")
    if not real:
        plt.plot(x, df["apoyo_latente_simulado"] * 100, "--", color="black", label="Apoyo latente (simulado)")
    plt.axhline(50, color="0.4", lw=0.6, ls=":")
    plt.xticks(list(x)[::3], [v[-5:] for v in df["ventana"]][::3], fontsize=7)
    plt.ylabel(f"% votos válidos: {CANDIDATOS[0]['nombre'].split(' (')[0]}")
    plt.xlabel("Hora (flujo simulado)" if not real else "Hora")
    plt.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "prediccion_tiempo_real.png"), dpi=200)


if __name__ == "__main__":
    main()
