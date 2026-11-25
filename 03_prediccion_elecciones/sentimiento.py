"""
Clasificador de sentimiento para publicaciones políticas en español.

Orden de preferencia:
1. pysentimiento (modelo RoBERTuito, entrenado con tweets en español). Es el modelo recomendado
   para producción: pip install pysentimiento  (descarga los pesos desde Hugging Face).
2. Respaldo local: TF-IDF de palabras y caracteres + regresión logística, entrenado con el
   corpus semilla de este archivo. Se usa cuando no hay acceso a Hugging Face.

El resultado de clasificar() es un diccionario {"POS": p, "NEG": p, "NEU": p}.
"""
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

# Plantillas etiquetadas; {c} se reemplaza por el nombre o hashtag de cualquier candidato,
# de modo que el modelo aprende el sentimiento y no a qué candidato se refiere.
POSITIVAS = [
    "Mi voto va para {c}", "Este 7 de junio voto por {c}", "{c} es la mejor opción para el país",
    "Confío en {c} para sacar adelante al Perú", "Excelente propuesta de {c} sobre seguridad",
    "{c} ganó el debate sin duda", "Apoyo total a {c}", "Con {c} el Perú sale adelante",
    "Gran mitin de {c} hoy, mucha gente", "{c} tiene el mejor plan económico",
    "Por fin un candidato que escucha a las regiones: {c}", "Orgulloso de votar por {c}",
    "{c} presidente, vamos con todo", "Me convenció {c} con sus propuestas", "Buen discurso de {c}",
    "{c} es el cambio que necesitamos", "Toda mi familia vota por {c}", "Vamos {c}, a ganar esta segunda vuelta",
    "{c} demostró experiencia y liderazgo", "Mi apoyo es para {c}, sin dudarlo",
    "Lo mejor del debate fue {c}", "Con {c} hay esperanza", "{c} habló claro sobre empleo, me gustó",
    "Ya decidí, voto {c}", "{c} merece la oportunidad", "Muy buena entrevista de {c}",
]
NEGATIVAS = [
    "Jamás votaría por {c}", "{c} es un peligro para el país", "No confío en {c}",
    "{c} perdió el debate, puras promesas", "Pésima propuesta de {c}", "Con {c} el país se hunde",
    "{c} no tiene plan, solo discursos", "Otra mentira de {c}", "Voto en contra de {c}",
    "{c} no representa a nadie", "Qué vergüenza lo que dijo {c}", "{c} es más de lo mismo",
    "No a {c}, nunca más", "Ni loco voto por {c}", "{c} esquivó las preguntas, muy mal",
    "Decepcionado de {c}", "{c} solo busca poder", "El peor candidato es {c}", "{c} miente sin parar",
    "Cuidado con {c}, no le crean", "Terrible entrevista de {c}", "{c} no sabe gobernar",
    "Antes viciado que votar por {c}", "{c} nos quiere engañar", "Malas noticias si gana {c}",
    "{c} improvisa en todo",
]
NEUTRAS = [
    "{c} presentará su plan de gobierno mañana", "Hoy entrevistan a {c} a las 8 pm",
    "{c} estará en Trujillo este jueves", "¿Alguien vio el debate de {c}?", "Resumen del día de {c}",
    "{c} se reunió con dirigentes regionales", "Así fue la conferencia de prensa de {c}",
    "{c} visitará Arequipa", "Encuesta: ¿qué opinan de {c}?", "{c} publicó su hoja de vida en el JNE",
    "Agenda de {c} para esta semana", "{c} respondió preguntas de la prensa", "Transmisión en vivo con {c}",
    "{c} habló sobre minería y agua", "Mañana es el debate entre {c} y su rival",
    "Nota de prensa del equipo de {c}", "{c} llegó al local de votación", "Datos del plan de {c} sobre salud",
]
NOMBRES_GENERICOS = ["la candidata", "el candidato", "el partido", "@candidato", "#Candidato2026",
                     "Candidato A", "Candidato B", "la fórmula presidencial"]


LEX_POS = {"voto", "votar", "apoyo", "confío", "confio", "mejor", "excelente", "buena", "buen", "bueno", "gran",
           "ganó", "gano", "esperanza", "convenció", "convencio", "orgulloso", "vamos", "merece", "liderazgo",
           "experiencia", "gustó", "gusto", "adelante", "decidí", "decidi", "cambio", "claro", "escucha", "total"}
LEX_NEG = {"jamás", "jamas", "nunca", "peligro", "peor", "pésima", "pesima", "terrible", "mentira", "miente",
           "vergüenza", "verguenza", "decepcionado", "engañar", "enganar", "hunde", "perdió", "perdio", "malas",
           "mal", "contra", "viciado", "cuidado", "improvisa", "esquivó", "esquivo", "poder", "promesas", "loco"}
NEGADORES = {"no", "ni", "nunca", "jamás", "jamas", "tampoco"}


class Lexico:
    """Rasgos de léxico: conteo de palabras positivas y negativas, considerando negaciones."""

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self

    def transform(self, X):
        filas = []
        for t in X:
            toks = re.findall(r"\w+", t.lower())
            pos = neg = 0
            for i, w in enumerate(toks):
                negado = any(x in NEGADORES for x in toks[max(0, i - 2):i])
                if w in LEX_POS:
                    neg, pos = (neg + 1, pos) if negado else (neg, pos + 1)
                elif w in LEX_NEG and w not in NEGADORES:
                    neg += 1
            hay_neg = int(any(w in NEGADORES for w in toks))
            filas.append([pos, neg, hay_neg, pos - neg])
        return np.array(filas, dtype=float)


def _limpiar(t):
    t = t.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " @usuario ", t)
    return re.sub(r"\s+", " ", t).strip()


def _corpus(semilla=7):
    r = random.Random(semilla)
    textos, etiquetas = [], []
    for lista, et in [(POSITIVAS, "POS"), (NEGATIVAS, "NEG"), (NEUTRAS, "NEU")]:
        for plantilla in lista:
            for _ in range(6):
                t = plantilla.format(c=r.choice(NOMBRES_GENERICOS))
                if r.random() < 0.3:
                    t += r.choice(["!!", " 🇵🇪", " #Elecciones2026", "...", " #SegundaVuelta"])
                textos.append(t)
                etiquetas.append(et)
    return textos, etiquetas


class AnalizadorSentimiento:
    def __init__(self, usar_transformer=True):
        self.motor = None
        if usar_transformer:
            try:
                from pysentimiento import create_analyzer
                self.motor = create_analyzer(task="sentiment", lang="es")
                self.nombre = "pysentimiento (RoBERTuito)"
            except Exception:
                self.motor = None
        if self.motor is None:
            textos, etiquetas = _corpus()
            vec = FeatureUnion([
                ("palabras", TfidfVectorizer(preprocessor=_limpiar, ngram_range=(1, 2), min_df=1)),
                ("caracteres", TfidfVectorizer(preprocessor=_limpiar, analyzer="char_wb", ngram_range=(3, 5))),
                ("lexico", Lexico()),
            ])
            self.clf = make_pipeline(vec, LogisticRegression(max_iter=2000, C=4))
            self.clf.fit(textos, etiquetas)
            self.nombre = "TF-IDF + regresión logística (respaldo local)"

    def clasificar(self, textos):
        if self.motor is not None:
            res = self.motor.predict(textos)
            return [dict(r.probas) for r in res]
        P = self.clf.predict_proba(textos)
        clases = list(self.clf.classes_)
        return [{c: float(p[clases.index(c)]) for c in ("POS", "NEG", "NEU")} for p in P]


def validar():
    """Validación cruzada del respaldo local con plantillas no vistas (separadas por plantilla)."""
    from sklearn.model_selection import GroupKFold, cross_val_score
    r = random.Random(11)
    textos, etiquetas, grupos = [], [], []
    for gi, (lista, et) in enumerate([(POSITIVAS, "POS"), (NEGATIVAS, "NEG"), (NEUTRAS, "NEU")]):
        for k, plantilla in enumerate(lista):
            for _ in range(4):
                textos.append(plantilla.format(c=r.choice(NOMBRES_GENERICOS)))
                etiquetas.append(et)
                grupos.append(f"{gi}-{k}")
    a = AnalizadorSentimiento(usar_transformer=False)
    s = cross_val_score(a.clf, textos, etiquetas, groups=grupos, cv=GroupKFold(5), scoring="f1_macro")
    return float(np.mean(s))


if __name__ == "__main__":
    a = AnalizadorSentimiento(usar_transformer=False)
    print("Modelo:", a.nombre)
    print("F1 macro (plantillas no vistas):", round(validar(), 3))
    for t, p in zip(["Voto por la candidata, gran propuesta", "Nunca votaría por ese candidato",
                     "Mañana es el debate"], a.clasificar(["Voto por la candidata, gran propuesta",
                                                           "Nunca votaría por ese candidato",
                                                           "Mañana es el debate"])):
        print(t, "->", {k: round(v, 2) for k, v in p.items()})
