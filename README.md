# Financiera Confianza Perú, modelos de IA y predicción electoral

Trabajo individual del curso Sistema de Información Integrados (UPAO, 2026).
Estudiante: De Lama Cespedes, Sergio Felix.

| Carpeta | Contenido |
|---------|-----------|
| `01_ml_scoring_crediticio/` | Ejemplo de machine learning: scoring crediticio con XGBoost frente a regresión logística (datos sintéticos) |
| `02_dl_cnn_digitos/` | Ejemplo de deep learning: red convolucional (Keras) para dígitos manuscritos |
| `03_prediccion_elecciones/` | Predicción en tiempo real de la segunda vuelta 2026 con análisis de sentimiento y actualización bayesiana |
| `04_iso25010_ux/` | Pantalla POS del Módulo Comercial de RestoIA evaluada con ISO/IEC 25010:2023 |
| `Financiera_IA_Elecciones_DeLama.pdf` | Documento con la solución completa |

## Ejecución

```bash
pip install -r requirements.txt
python 01_ml_scoring_crediticio/scoring_crediticio.py
KERAS_BACKEND=jax python 02_dl_cnn_digitos/cnn_digitos.py
cd 03_prediccion_elecciones && python demo_tiempo_real.py        # flujo simulado
cd 03_prediccion_elecciones && python demo_tiempo_real.py --x    # API de X (requiere X_BEARER_TOKEN)
```

Los resultados (gráficas y métricas) se guardan en la carpeta `resultados/` de cada programa.
El flujo de `demo_tiempo_real.py` sin `--x` es simulado: valida el algoritmo, no predice el resultado real.
