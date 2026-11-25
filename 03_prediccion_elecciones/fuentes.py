"""
Fuentes de publicaciones en tiempo real.

- FlujoX: conecta con el endpoint de flujo filtrado de la API v2 de X (antes Twitter).
  Requiere una cuenta de desarrollador con acceso al stream y la variable de entorno
  X_BEARER_TOKEN. Las reglas filtran por nombres y hashtags de los candidatos.
- SimuladorRedes: genera publicaciones SINTÉTICAS para probar el sistema sin acceso a la API.
  El apoyo de cada candidato es un parámetro conocido del simulador, lo que permite medir si el
  predictor lo recupera. Sus resultados no son una predicción de la elección real.
"""
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from sentimiento import NEGATIVAS, NEUTRAS, POSITIVAS


@dataclass
class Publicacion:
    id: str
    autor: str
    texto: str
    fecha: datetime
    seguidores: int = 100
    antiguedad_cuenta_dias: int = 365


class FlujoX:
    URL = "https://api.x.com/2/tweets/search/stream"
    URL_REGLAS = "https://api.x.com/2/tweets/search/stream/rules"

    def __init__(self, candidatos):
        import requests
        self.requests = requests
        self.token = os.environ["X_BEARER_TOKEN"]
        self.candidatos = candidatos

    def _h(self):
        return {"Authorization": f"Bearer {self.token}"}

    def configurar_reglas(self):
        reglas = [{"value": f'({" OR ".join(c["terminos"])}) lang:es -is:retweet', "tag": c["id"]}
                  for c in self.candidatos]
        self.requests.post(self.URL_REGLAS, headers=self._h(), json={"add": reglas}, timeout=30)

    def __iter__(self):
        params = {"tweet.fields": "created_at,author_id", "expansions": "author_id",
                  "user.fields": "created_at,public_metrics"}
        with self.requests.get(self.URL, headers=self._h(), params=params, stream=True, timeout=90) as r:
            for linea in r.iter_lines():
                if not linea:
                    continue
                d = json.loads(linea)
                t, u = d["data"], d.get("includes", {}).get("users", [{}])[0]
                creada = datetime.fromisoformat(u.get("created_at", t["created_at"]).replace("Z", "+00:00"))
                yield Publicacion(t["id"], t["author_id"], t["text"],
                                  datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")),
                                  u.get("public_metrics", {}).get("followers_count", 0),
                                  (datetime.now(creada.tzinfo) - creada).days)


class SimuladorRedes:
    """Genera un flujo de publicaciones con un apoyo latente que oscila alrededor de un empate."""

    def __init__(self, candidatos, apoyo_base=0.50, amplitud=0.03, horas=24, pubs_por_hora=400,
                 proporcion_bots=0.08, semilla=2026):
        self.c = candidatos
        self.base, self.amp = apoyo_base, amplitud
        self.horas, self.n, self.bots = horas, pubs_por_hora, proporcion_bots
        self.r = random.Random(semilla)
        self.inicio = datetime(2026, 6, 6, 0, 0)
        self.cuentas_bot = [f"bot_{k}" for k in range(25)]

    def apoyo_latente(self, h):
        """Apoyo real (desconocido para el predictor) del primer candidato en la hora h."""
        return self.base + self.amp * math.sin(2 * math.pi * h / self.horas)

    def __iter__(self):
        k = 0
        for h in range(self.horas):
            pA = self.apoyo_latente(h)
            for _ in range(self.n):
                k += 1
                fecha = self.inicio + timedelta(hours=h, seconds=self.r.randint(0, 3599))
                if self.r.random() < self.bots:        # cuentas automatizadas que inflan a un candidato
                    obj = self.r.choice(self.c)
                    texto = self.r.choice(POSITIVAS).format(c=self.r.choice(obj["terminos"]))
                    yield Publicacion(f"p{k}", self.r.choice(self.cuentas_bot), texto, fecha, 3, 4)
                    continue
                simpatiza_A = self.r.random() < pA
                propio, rival = (self.c[0], self.c[1]) if simpatiza_A else (self.c[1], self.c[0])
                u = self.r.random()
                if u < 0.55:
                    texto = self.r.choice(POSITIVAS).format(c=self.r.choice(propio["terminos"]))
                elif u < 0.85:
                    texto = self.r.choice(NEGATIVAS).format(c=self.r.choice(rival["terminos"]))
                else:
                    texto = self.r.choice(NEUTRAS).format(c=self.r.choice(self.r.choice(self.c)["terminos"]))
                yield Publicacion(f"p{k}", f"u{self.r.randint(1, 60000)}", texto, fecha,
                                  self.r.randint(10, 3000), self.r.randint(60, 4000))
