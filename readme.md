# Consejero de Zagreus

Asistente documental sobre el videojuego **Hades (2018)** de Supergiant Games. Responde preguntas buscando antes en una base de documentos y usando solo lo que encuentra (técnica **RAG**: *Retrieval-Augmented Generation*). Siempre muestra de qué fragmentos ha sacado la respuesta y, si la información no está en los documentos, lo dice en vez de inventar.

> Proyecto final "Diseña tu propio asistente".

## Cómo funciona

```
Pregunta → embedding → búsqueda (significado + palabras) → 5 fragmentos → prompt → Qwen → respuesta + fuentes
```

1. **Base documental:** ~1.900 fragmentos de la wiki de Hades en Fandom (`data/hades1.json`) más las builds escritas por el grupo (`data/builds.json`). Se quitan las frases sobre Hades II, los fragmentos muy cortos y los repetidos.
2. **Embeddings:** cada fragmento se convierte en un vector de 768 números con el modelo `BAAI/bge-base-en-v1.5`.
3. **Qdrant:** base de datos de vectores donde se guardan los fragmentos.
4. **Recuperación híbrida:** se busca de dos formas, por significado (Qdrant) y por palabras exactas (BM25), y se juntan los resultados con RRF. Se devuelven los 5 mejores.
5. **Generación:** el modelo `qwen2.5:3b` (ejecutado con Ollama) responde usando solo esos fragmentos y cita sus ids, por ejemplo `[58]`.
6. **Memoria:** si la pregunta depende de la anterior ("¿y qué hace el de Hera?"), se reescribe para que se entienda sola antes de buscar.

## Archivos del proyecto

| Archivo | Para qué sirve |
|---|---|
| `RAG_Hades1.ipynb` | Notebook para **Google Colab** con todo el desarrollo, las pruebas y la evaluación |
| `asistente.py` | Versión para **terminal**: haces preguntas con `input()` y responde |
| `data/hades1.json` | Fragmentos de la wiki de Hades |
| `data/builds.json` | Builds escritas por el grupo |
| `embeddings.npy` | Vectores calculados (se genera solo la primera vez) |

## Instalación para usar el asistente por terminal

No hace falta GPU. Se ha probado en Windows con 16 GB de RAM y Python 3.11.

**Requisitos:** Python 3.9 o superior, unos 4 GB libres en disco, 8 GB de RAM como mínimo (mejor 16 GB) e internet la primera vez.

### 1. Instalar Ollama

- **Windows** (PowerShell): `irm https://ollama.com/install.ps1 | iex`
- **Mac:** descargar la aplicación desde https://ollama.com/download
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

Cierra y vuelve a abrir la terminal y comprueba que funciona:

```
ollama --version
```

### 2. Descargar el modelo

```
ollama pull qwen2.5:3b
```

Ocupa unos 2 GB.

### 3. Preparar el proyecto

Abre la carpeta del proyecto en VS Code y, en su terminal, crea un entorno de Python:

```
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Cuando esté activo, la línea de la terminal empezará por `(.venv)`. Si en Windows da error de scripts deshabilitados, ejecuta antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### 4. Instalar las librerías

```
pip install sentence-transformers qdrant-client rank-bm25 numpy ollama
```

### 5. Ejecutar

```
python asistente.py
```

- **La primera vez** descarga el modelo de embeddings (~440 MB) y calcula los vectores (unos 3 minutos sin GPU). Se guardan en `embeddings.npy`, así que las siguientes veces arranca en pocos segundos.
- Si cambias `data/builds.json` o `data/hades1.json`, **borra `embeddings.npy`** para que se recalculen.

## Cómo usarlo

Haz las preguntas **en inglés**, porque los documentos están en inglés.

```
You: Which aspects does the Heart-Seeking Bow have?
You: And what does the one named after Hera do?
You: How many copies of Hades has Supergiant Games sold worldwide?
```

La primera pregunta devuelve los 4 aspectos con sus fuentes, la segunda prueba la memoria y la tercera no está en los documentos, así que responde `I can't find that information in the available documents.`

| Comando | Qué hace |
|---|---|
| `nuevo` | Borra la memoria de la conversación |
| `salir` | Termina el programa |

## Notebook de Colab

`RAG_Hades1.ipynb` contiene el mismo sistema usando una GPU y el modelo `Qwen/Qwen2.5-3B-Instruct` sin comprimir, además de la evaluación:

1. Abrir el notebook en Google Colab y activar la GPU (Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU).
2. Subir `hades1.json` y `builds.json` cuando lo pida.
3. Ejecutar todas las celdas de arriba abajo.

## Problemas frecuentes

| Problema | Solución |
|---|---|
| `ollama` no se reconoce como comando | Cerrar y volver a abrir la terminal |
| Error de conexión con Ollama | Abrir la aplicación Ollama (tiene que estar ejecutándose) |
| Avisos sobre *symlinks* o `HF_TOKEN` | Son normales, se pueden ignorar |
| Cada respuesta tarda | Normal sin GPU: el modelo se ejecuta en CPU (unos segundos por respuesta) |

## Limitaciones conocidas

- Los documentos y las preguntas están en inglés.
- La búsqueda híbrida no mejoró a la búsqueda solo por significado en las pruebas (hit@5: 0.92 frente a 1.00). La memoria sí mejoró el sistema.
- A veces el modelo cita un fragmento que no contiene exactamente lo que afirma, o cambia algún detalle del texto original, por lo que conviene comprobar las fuentes.
- La similitud sola no permite decidir cuándo callarse; la abstención depende de la instrucción del prompt.
- La memoria reescribe las preguntas con un modelo pequeño, que puede perder matices.

## Datos

Los textos proceden de la wiki de Hades en Fandom (https://hades.fandom.com). Cada fragmento conserva la URL de la página de origen.