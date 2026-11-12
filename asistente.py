import os, re, json

CONFIG = {
    "ruta_hades": "data/hades1.json",
    "ruta_builds": "data/builds.json",
    "modelo_embeddings": "BAAI/bge-base-en-v1.5",
    "prefijo_consulta": "Represent this sentence for searching relevant passages: ",
    "modelo_llm": "qwen2.5:3b",
    "coleccion": "hades1",
    "k": 5,
    "turnos_memoria": 4,
}
NO_INFO = "I can't find that information in the available documents."

PATRON_HADES2 = re.compile(r"Melino[eë]|Hades II", re.IGNORECASE)


def quitar_hades2(doc):
    original = doc["texto"].strip()
    lineas = []
    for linea in original.split("\n"):
        frases = re.split(r"(?<=[.!?])\s+", linea)
        frases = [f for f in frases if not PATRON_HADES2.search(f)]
        if frases:
            lineas.append(" ".join(frases))
    limpio = "\n".join(lineas).strip()
    # si se perdió la cabecera "Entidad - Sección:", se restituye
    if limpio and limpio != original and not limpio.startswith(doc["entidad"]):
        limpio = f"{doc['entidad']} - {doc.get('seccion', '')}: {limpio}"
    return limpio


def cargar_documentos():
    with open(CONFIG["ruta_hades"], encoding="utf-8") as f:
        crudos = json.load(f)

    if os.path.exists(CONFIG["ruta_builds"]):
        with open(CONFIG["ruta_builds"], encoding="utf-8") as f:
            for b in json.load(f):
                if str(b.get("texto", "")).strip().startswith("["):
                    continue
                b.setdefault("entidad", "Build")
                b.setdefault("categoria", "build")
                b.setdefault("tipo", "juego")
                b.setdefault("seccion", "Build")
                b.setdefault("fuente", "grupo (propio)")
                b.setdefault("url", "")
                crudos.append(b)

    docs, vistos, descartados = [], set(), 0
    for d in crudos:
        texto = quitar_hades2(d)
        if len(texto.split()) < 12 or texto in vistos:
            descartados += 1
            continue
        vistos.add(texto)
        docs.append({
            "id": len(docs),
            "entidad": d["entidad"],
            "seccion": d.get("seccion", ""),
            "categoria": d.get("categoria", "general"),
            "tipo": d.get("tipo", "juego"),
            "fuente": d.get("fuente", ""),
            "url": d.get("url", ""),
            "texto": texto,
        })
    print(f"Fragmentos cargados: {len(docs)} (descartados: {descartados})")
    return docs


DOCS = cargar_documentos()

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

RUTA_EMB = "embeddings.npy"

print("Cargando modelo de embeddings...")
modelo_embeddings = SentenceTransformer(CONFIG["modelo_embeddings"])


def calcular_embeddings():
    if os.path.exists(RUTA_EMB):
        emb = np.load(RUTA_EMB)
        if emb.shape[0] == len(DOCS):
            print("Embeddings cargados desde", RUTA_EMB)
            return emb
    print("Calculando embeddings (la primera vez tarda unos minutos)...")
    emb = modelo_embeddings.encode(
        [d["texto"] for d in DOCS],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    np.save(RUTA_EMB, emb)
    return emb


EMB = calcular_embeddings()
print("Matriz de embeddings:", EMB.shape)


def embedding_pregunta(pregunta):
    return modelo_embeddings.encode(
        CONFIG["prefijo_consulta"] + pregunta,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


cliente_qdrant = QdrantClient(":memory:")
cliente_qdrant.create_collection(
    collection_name=CONFIG["coleccion"],
    vectors_config=models.VectorParams(size=EMB.shape[1], distance=models.Distance.COSINE),
)

for inicio in range(0, len(DOCS), 256):
    lote = range(inicio, min(inicio + 256, len(DOCS)))
    cliente_qdrant.upsert(
        collection_name=CONFIG["coleccion"],
        points=[models.PointStruct(id=i, vector=EMB[i].tolist(), payload=DOCS[i]) for i in lote],
    )

print("Puntos en Qdrant:", cliente_qdrant.count(collection_name=CONFIG["coleccion"]).count)