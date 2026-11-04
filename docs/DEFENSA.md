# Preguntas de defensa

Cada fase agrega 3–5 preguntas con una respuesta corta. Cualquier integrante debe poder responder cualquiera.

## Fase 1 — Idea de negocio y CRISP-DM

**¿Por qué F1 de la clase premium y no la exactitud?**
Solo cerca del 14 % de los tintos son premium **[Verificar]**. Un modelo que siempre dijera "no premium" tendría ~86 % de exactitud y F1 igual a cero. El F1 equilibra precisión y recall de la clase que le importa a la bodega (D-04).

**¿Por qué se entrena solo con tintos y qué hacen los blancos?**
El escenario de negocio es una bodega de tintos que lanza una línea de blancos y reutiliza el modelo. Los blancos tienen otra distribución y sirven como lote de data drift (D-02). No sirven para concept drift, porque ahí cambian también las entradas.

**¿Por qué se deduplica antes de dividir?**
Si se divide primero, filas idénticas pueden caer en `train` y `test`, y las métricas de prueba quedan infladas por fuga de datos (D-03).

**¿Con qué conjunto se elige el modelo y con cuál se mide?**
La familia, los hiperparámetros y el umbral de decisión se eligen con la validación cruzada sobre `train`. `test` se mide una sola vez, para que los criterios de negocio (recall ≥ 0,70, precisión ≥ 0,50) y el `F1_ref` del concept drift no queden optimistas.

**¿Por qué importan los ~1.000 registros mínimos si solo se entrena con tintos?**
La consigna pide al menos 1.000 registros. El dataset completo tiene 6.497 y el subconjunto de tintos tiene 1.599 (algo menos tras deduplicar **[Verificar]**), así que ambos cumplen.

## Fase 4 — API y Model Registry

**¿Cómo sabe la API qué modelo servir?**
Lee `MODEL_URI` y carga el alias `champion` desde MLflow al arrancar; obtiene versión y `run_id` del registro.

**¿Qué diferencia hay entre `/health` y `/ready`?**
`/health` confirma que el proceso vive. `/ready` responde 200 solo cuando el modelo está cargado; de lo contrario devuelve 503.

**¿Para qué sirve `served_by`?**
Identifica el pod o contenedor que atendió la petición y permite demostrar el balanceo entre réplicas.

## Fase 5 — Docker

**¿Qué contiene la imagen de la API?**
Python, dependencias fijadas, el código de `api/` y `src/wine/config.py`; el modelo se carga desde MLflow.

**¿Por qué `depends_on` espera el healthcheck de MLflow?**
La API necesita que el servidor responda antes de intentar resolver el alias `champion` y descargar el modelo.

**¿Qué demuestra la prueba de Compose?**
Que los contenedores responden `/ready` y `/predict` sin activar un entorno Python local, usando el modelo real versión 3.
