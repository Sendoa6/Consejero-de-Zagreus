# =====================================================
# 1. LIBRERÍAS Y AJUSTES
# =====================================================
import os, re, json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

RUTA_HADES = "data/hades1.json"
RUTA_BUILDS = "data/builds.json"
RUTA_EMB = "embeddings.npy"
MODELO_EMB = "BAAI/bge-base-en-v1.5"                 # convierte textos en vectores
PREFIJO = "Represent this sentence for searching relevant passages: "
MODELO_LLM = "qwen2.5:3b"                            # modelo que redacta (en Ollama)
K = 5                                                # fragmentos que se recuperan
TURNOS = 4                                           # turnos de memoria
NO_INFO = "I can't find that information in the available documents."


# =====================================================
# 2. CARGAR Y LIMPIAR LOS DOCUMENTOS
# =====================================================
def quitar_hades2(doc):
    """Borra las frases que hablan de Hades II o Melinoë."""
    patron = re.compile(r"Melino[eë]|Hades II", re.IGNORECASE)
    original = doc["texto"].strip()
    lineas = []
    for linea in original.split("\n"):
        frases = re.split(r"(?<=[.!?])\s+", linea)
        frases = [f for f in frases if not patron.search(f)]
        if frases:
            lineas.append(" ".join(frases))
    limpio = "\n".join(lineas).strip()
    if limpio and limpio != original and not limpio.startswith(doc["entidad"]):
        limpio = f"{doc['entidad']} - {doc.get('seccion', '')}: {limpio}"
    return limpio


def cargar_documentos():
    with open(RUTA_HADES, encoding="utf-8") as f:
        crudos = json.load(f)

    # Añadimos las builds del grupo (si existe el archivo)
    if os.path.exists(RUTA_BUILDS):
        valores_por_defecto = {"entidad": "Build", "tipo": "juego", "seccion": "Build", "url": ""}
        with open(RUTA_BUILDS, encoding="utf-8") as f:
            for b in json.load(f):
                if str(b.get("texto", "")).strip().startswith("["):   # plantilla sin rellenar
                    continue
                for clave, valor in valores_por_defecto.items():
                    b.setdefault(clave, valor)
                crudos.append(b)

    # Limpiamos: fuera fragmentos cortos y repetidos
    docs, vistos = [], set()
    for d in crudos:
        texto = quitar_hades2(d)
        if len(texto.split()) < 12 or texto in vistos:
            continue
        vistos.add(texto)
        docs.append({"id": len(docs), "entidad": d["entidad"], "seccion": d.get("seccion", ""),
                     "tipo": d.get("tipo", "juego"), "url": d.get("url", ""), "texto": texto})
    return docs


DOCS = cargar_documentos()
print("Fragmentos cargados:", len(DOCS))


# =====================================================
# 3. EMBEDDINGS Y BASE DE DATOS QDRANT
# =====================================================
print("Cargando modelo de embeddings...")
modelo_emb = SentenceTransformer(MODELO_EMB)

# Si ya calculamos los vectores otra vez, los cargamos; si no, los calculamos y guardamos
if os.path.exists(RUTA_EMB) and np.load(RUTA_EMB).shape[0] == len(DOCS):
    EMB = np.load(RUTA_EMB)
else:
    print("Calculando embeddings (la primera vez tarda unos minutos)...")
    EMB = modelo_emb.encode([d["texto"] for d in DOCS], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=True)
    np.save(RUTA_EMB, EMB)

# Guardamos los vectores en Qdrant (en memoria), junto con los datos de cada fragmento
qdrant = QdrantClient(":memory:")
qdrant.create_collection("hades1", vectors_config=models.VectorParams(
    size=EMB.shape[1], distance=models.Distance.COSINE))
qdrant.upsert("hades1", points=[
    models.PointStruct(id=d["id"], vector=EMB[d["id"]].tolist(), payload=d) for d in DOCS])


# =====================================================
# 4. RECUPERACIÓN: BUSCAR LOS FRAGMENTOS MÁS ÚTILES
# =====================================================
def tokenizar(texto):
    return re.findall(r"[a-z0-9']+", texto.lower())


BM25 = BM25Okapi([tokenizar(d["texto"]) for d in DOCS])        # búsqueda por palabras
ES_JUEGO = np.array([d["tipo"] == "juego" for d in DOCS])      # solo buscamos en "juego"
SOLO_JUEGO = models.Filter(must=[models.FieldCondition(key="tipo", match=models.MatchValue(value="juego"))])


def recuperar(pregunta):
    vector = modelo_emb.encode(PREFIJO + pregunta, normalize_embeddings=True)

    # Búsqueda 1: por significado (vectores en Qdrant)
    respuesta = qdrant.query_points("hades1", query=vector.tolist(), limit=30,
                                    query_filter=SOLO_JUEGO, with_payload=False)
    por_significado = [p.id for p in respuesta.points]

    # Búsqueda 2: por palabras exactas (BM25)
    puntos = np.where(ES_JUEGO, BM25.get_scores(tokenizar(pregunta)), -1.0)
    orden = np.argsort(puntos)[::-1][:30]
    por_palabras = [int(i) for i in orden if puntos[i] > 0]

    # Juntamos las dos listas: puntúa más quien sale arriba en ambas (RRF)
    acumulado = {}
    for lista in (por_significado, por_palabras):
        for posicion, i in enumerate(lista):
            acumulado[i] = acumulado.get(i, 0) + 1 / (60 + posicion + 1)
    mejores = sorted(acumulado, key=acumulado.get, reverse=True)[:K]

    return [(DOCS[i], float(EMB[i] @ vector)) for i in mejores]    # (fragmento, similitud)


# =====================================================
# 5. GENERACIÓN: EL MODELO REDACTA LA RESPUESTA
# =====================================================
SYSTEM_RAG = (
    "You are an expert assistant for the video game Hades (2018) by Supergiant Games. "
    "Answer the question using ONLY the numbered CONTEXT passages. "
    f"If the context does not contain the answer, reply exactly: {NO_INFO} "
    "Do not use outside knowledge and do not talk about Hades II. "
    "Be concise (at most 5 sentences) and cite the ids of the passages you used, like [12] [40]."
)

SYSTEM_REESCRIBIR = (
    "Rewrite the user's last question as a standalone question, using the conversation to resolve "
    "pronouns and references such as 'it', 'that one' or 'its'. "
    "Keep exactly the same meaning: do not change what is being asked and do not add new topics. "
    "Keep game names exactly as written. "
    "If the question is already clear on its own, repeat it unchanged. "
    "Output only the question and nothing else."
)

# Palabras que indican que la pregunta depende de la anterior
PALABRAS_REFERENCIA = {"it", "its", "that", "this", "those", "these", "they", "them", "their",
                       "he", "she", "his", "her", "one", "and", "also", "same", "other"}


def necesita_reescribir(pregunta):
    return len(set(tokenizar(pregunta)) & PALABRAS_REFERENCIA) > 0


def generar(mensajes, max_tokens=300):
    r = ollama.chat(model=MODELO_LLM, messages=mensajes,
                    options={"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096})
    return r["message"]["content"].strip()


def responder(pregunta, historial):
    recientes = historial[-TURNOS:]

    # Memoria: si hay conversación previa, reescribimos la pregunta para que se entienda sola
    autonoma = pregunta
    if recientes and necesita_reescribir(pregunta):
        conversacion = "\n".join(f"User: {p}\nAssistant: {r}" for p, r in recientes)
        nueva = generar([{"role": "system", "content": SYSTEM_REESCRIBIR},
                         {"role": "user", "content": f"Conversation:\n{conversacion}\n\nLast question: {pregunta}"}],
                        max_tokens=60).split("\n")[0].strip()
        if 5 <= len(nueva) <= 300:
            autonoma = nueva

    # Buscamos los fragmentos y se los damos al modelo como CONTEXT
    fragmentos = recuperar(autonoma)
    contexto = "\n\n".join(f"[{d['id']}] {d['texto']}" for d, _ in fragmentos)
    mensajes = [{"role": "system", "content": SYSTEM_RAG}]
    for p, r in recientes:
        mensajes += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
    mensajes.append({"role": "user", "content": f"CONTEXT:\n{contexto}\n\nQUESTION: {autonoma}"})

    respuesta = generar(mensajes)
    historial.append((pregunta, respuesta))
    return autonoma, respuesta, fragmentos


# =====================================================
# 6. BUCLE CON input() (lo que se ejecuta al lanzar el programa)
# =====================================================
def chat():
    historial = []
    print("\n=== Consejero de Zagreus ===")
    print("Pregunta sobre Hades (en inglés). Escribe 'nuevo' para borrar la memoria o 'salir' para terminar.")
    while True:
        pregunta = input("\nYou: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        if pregunta.lower() == "nuevo":
            historial.clear()
            print("Conversación reiniciada.")
            continue

        autonoma, respuesta, fragmentos = responder(pregunta, historial)
        print("\nANSWER:", respuesta)
        if autonoma != pregunta:
            print("(pregunta reescrita:", autonoma + ")")
        print("SOURCES:")
        for d, similitud in fragmentos:
            print(f"  [{d['id']}] {d['entidad']} - {d['seccion']} ({similitud:.3f}) {d['url']}")


if __name__ == "__main__":
    chat()