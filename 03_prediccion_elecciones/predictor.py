"""
Predictor de intención de voto en tiempo real a partir de redes sociales.

Por cada ventana de tiempo (por defecto, una hora):
1. Asigna cada publicación al candidato que menciona (se descartan las que mencionan a ambos).
2. Clasifica su sentimiento (POS, NEG, NEU).
3. Reduce el peso de cuentas sospechosas de ser bots (cuentas nuevas, pocos seguidores o
   que publican en exceso dentro de la ventana).
4. Convierte el sentimiento en una señal de apoyo: una mención positiva suma apoyo al candidato
   mencionado y una negativa suma apoyo a su rival (en una segunda vuelta solo hay dos opciones).
5. Actualiza una distribución Beta (bayesiana) con olvido exponencial, de modo que las ventanas
   recientes pesan más. De la Beta salen la estimación, su intervalo de credibilidad al 90 % y la
   probabilidad de que cada candidato supere el 50 % de votos válidos.
"""
import re
from collections import Counter, defaultdict

from scipy.stats import beta


class PredictorTiempoReal:
    def __init__(self, candidatos, analizador, prior=(0.5, 0.5), fuerza_prior=200, olvido=0.85,
                 max_pubs_cuenta=5):
        self.c = candidatos
        self.an = analizador
        self.alfa = fuerza_prior * prior[0]
        self.beta = fuerza_prior * prior[1]
        self.olvido = olvido
        self.max_pubs = max_pubs_cuenta
        self.patrones = {c["id"]: re.compile("|".join(re.escape(t.lower()) for t in c["terminos"]))
                         for c in candidatos}
        self.historial = []

    def _candidato(self, texto):
        t = texto.lower()
        hallados = [cid for cid, p in self.patrones.items() if p.search(t)]
        return hallados[0] if len(hallados) == 1 else None

    def _peso(self, pub, conteo):
        w = 1.0
        if pub.antiguedad_cuenta_dias < 30:
            w *= 0.2
        if pub.seguidores < 10:
            w *= 0.5
        if conteo[pub.autor] > self.max_pubs:
            w *= self.max_pubs / conteo[pub.autor]
        return w

    def actualizar(self, ventana, etiqueta):
        """Procesa una ventana de publicaciones y devuelve la estimación vigente."""
        A, B = self.c[0]["id"], self.c[1]["id"]
        conteo = Counter(p.autor for p in ventana)
        asignadas = [(p, self._candidato(p.texto)) for p in ventana]
        asignadas = [(p, cid) for p, cid in asignadas if cid]
        probs = self.an.clasificar([p.texto for p, _ in asignadas]) if asignadas else []
        apoyo = defaultdict(float)
        for (p, cid), pr in zip(asignadas, probs):
            w = self._peso(p, conteo)
            otro = B if cid == A else A
            apoyo[cid] += w * pr["POS"]
            apoyo[otro] += w * pr["NEG"]
        # olvido exponencial: lo anterior pierde peso antes de sumar la nueva evidencia
        self.alfa = self.alfa * self.olvido + apoyo[A]
        self.beta = self.beta * self.olvido + apoyo[B]
        est = self.alfa / (self.alfa + self.beta)
        li, ls = beta.ppf([0.05, 0.95], self.alfa, self.beta)
        fila = {"ventana": etiqueta, "publicaciones": len(ventana), "asignadas": len(asignadas),
                f"estimacion_{A}": est, "ic90_inf": li, "ic90_sup": ls,
                f"prob_gana_{A}": 1 - beta.cdf(0.5, self.alfa, self.beta)}
        self.historial.append(fila)
        return fila
