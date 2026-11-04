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

## Fase 2–3 — Datos, ETL, entrenamiento y drift

**¿Cómo se garantiza que los datos son los mismos en cada máquina?**
`download.py` fija el SHA-256 de cada CSV y la descarga falla si el origen cambia. Los splits se guardan con su propio SHA-256 en `checksums.json`, y dos ejecuciones del ETL dan los mismos hashes.

**¿Por qué el ETL tiene dos niveles de rango?**
Los rangos de tintos (por ejemplo `density` 0,98–1,01) protegen el entrenamiento. Si se aplicaran también a un lote de blancos, que llega a densidad 1,039, el pipeline abortaría antes de que el detector pudiera marcarlo ROJO. Por eso a los lotes de monitoreo solo se les exigen los límites de integridad.

**¿Cómo se eligió el modelo sin usar `test`?**
Con la validación cruzada de 5 folds sobre `train`: se escoge el mayor `cv_f1_mean`, y el umbral de decisión se ajusta con las predicciones fuera de fold. `test` se mide una sola vez. El ganador fue un Random Forest con umbral 0,22 (`cv_f1_mean` 0,568, `test_f1` 0,500).

**¿Cumple el modelo los criterios de negocio?**
Parcialmente. Recall 0,714 sí cumple (≥ 0,70), pero la precisión 0,385 (< 0,50) y el F1 0,500 (< 0,60) no. Por eso el `champion` tiene `validated=false` y la brecha se documenta como limitación en `crispdm.md` §5. La rúbrica evalúa la ingeniería, no la exactitud.

**¿Por qué la regla de data drift usa 3 de 11 variables?**
Se calibró con 200 lotes bootstrap de `monitor_green`: con `k = 3` hay 1 falso ROJO (0,5 %, por debajo del 5 % exigido) y se siguen detectando los blancos y el lote inyectado. Con `k ≥ 4` el lote inyectado, que desplaza exactamente tres variables, dejaría de detectarse.

**¿Por qué cambió el resultado del chi² al cambiar la referencia?**
Con `test` (204 filas) como referencia hubo 10,5 % de falsos ROJOS, porque su tasa predicha (25,5 %) difiere por azar de la de `monitor_green` (18,6 %). Con `train` (951 filas) hubo 0,5 %. Una referencia pequeña mete ruido en la comparación.

**¿Por qué el concept drift simulado no dispara la alarma?**
El modelo depende sobre todo de `alcohol` y `sulphates`, y la regla del nuevo panel marca premium a un subconjunto de los vinos que ya marca. Al reetiquetar, el recall sube de 0,71 a 0,95 y el F1 apenas cambia (0,50 → 0,53), así que no hay degradación que detectar. El mecanismo (umbral `F1_ref − 3σ`, 3 lotes consecutivos) se comprueba con un modelo oráculo.

## Fase 4 — API y Model Registry

**¿Cómo sabe la API qué modelo servir?**
Lee `MODEL_URI` y carga el alias `champion` desde MLflow al arrancar; obtiene versión y `run_id` del registro.

**¿Qué diferencia hay entre `/health` y `/ready`?**
`/health` confirma que el proceso vive. `/ready` responde 200 solo cuando el modelo está cargado; de lo contrario devuelve 503.

**¿Para qué sirve `served_by`?**
Identifica el pod o contenedor que atendió la petición y permite demostrar el balanceo entre réplicas.

**¿Por qué el modelo registrado devuelve probabilidades y no la clase?**
Porque el umbral de decisión depende del modelo (0,22 en el `champion`). El registro devuelve `[P(no premium), P(premium)]` y el umbral va en el tag `decision_threshold` de la versión; así la API puede aplicarlo y se puede cambiar sin reentrenar.

**¿Qué diferencia hay entre `champion` y `challenger`?**
`champion` es la versión que sirve la API (v3, Random Forest); `challenger` es el candidato (v2, regresión logística) y solo se promovería si supera al actual en el último lote etiquetado.

## Fase 5 — Docker

**¿Qué contiene la imagen de la API?**
Python, dependencias fijadas, el código de `api/` y `src/wine/config.py`; el modelo se carga desde MLflow.

**¿Por qué `depends_on` espera el healthcheck de MLflow?**
La API necesita que el servidor responda antes de intentar resolver el alias `champion` y descargar el modelo.

**¿Qué demuestra la prueba de Compose?**
Que los contenedores responden `/ready` y `/predict` sin activar un entorno Python local, usando el modelo real versión 3.

## Fase 6 — Kubernetes

**¿Por qué el Service es `LoadBalancer` y no `NodePort`?**
El Kubernetes de Docker Desktop usa nodos kind: contenedores en una red interna (172.18.0.0/16) que no se enruta desde macOS, así que `localhost:30080` no responde. Un `LoadBalancer` recibe IP externa y Docker Desktop lo publica en `localhost:8080`. El reparto entre pods lo sigue haciendo kube-proxy.

**¿Cómo se demuestra el balanceo y por qué `Connection: close`?**
Con 20 peticiones a `/predict` que devuelven `served_by`. Sin keep-alive cada petición abre una conexión nueva; si se reutilizara una sola, kube-proxy (que reparte por conexión) enviaría todo al mismo pod. El resultado fue 2 pods distintos, sin reparto exactamente parejo porque la elección es aleatoria.

**¿Qué pasa cuando se borra un pod?**
El ReplicaSet crea otro para volver al estado deseado. Sin fallos gracias a la readiness probe (solo recibe tráfico un pod con modelo cargado) y a `preStop: sleep 5` (el pod sale de los endpoints antes de cerrarse). La prueba registró 117 de 117 respuestas HTTP 200.

**¿Qué hace cada probe?**
`startupProbe` da hasta 150 s para cargar el modelo; `readinessProbe` (`/ready`) decide si recibe tráfico; `livenessProbe` (`/health`) reinicia el contenedor si se cuelga. Se subió el timeout a 5 s porque con 1 s fallaban bajo carga.

**¿Por qué solo 2 réplicas y no más?**
Cada réplica carga el modelo en memoria y Docker Desktop tiene 6 GB. Con 4 GB el plano de control de kind se reiniciaba. Dos réplicas bastan para demostrar balanceo y autorreparación; el escalado manual llega a 4 y el HPA llegará a 6.

## Fase 7 — Autoescalado (HPA)

**¿Cómo decide el HPA cuántos pods crear?**
Compara la CPU media de los pods con el objetivo (60 % de `requests.cpu`, o sea ~60 m por pod) y calcula réplicas = ceil(réplicas actuales × uso actual / objetivo), acotado entre `minReplicas` 2 y `maxReplicas` 6. En la prueba, con la CPU en 340 % pidió subir de 2 a 4 y luego a 6.

**¿Por qué hacen falta `requests` de CPU y metrics-server?**
El porcentaje se calcula sobre el request: sin él el HPA no puede medir ("unknown"). metrics-server es quien publica el uso de CPU de cada pod en la Metrics API; en Docker Desktop requirió `--kubelet-insecure-tls` por los certificados autofirmados.

**¿Por qué la subida es rápida y la bajada lenta?**
`scaleUp` agrega hasta 2 pods cada 30 s (cada pod carga el modelo en memoria). `scaleDown` espera una ventana de estabilización de 60 s y baja 1 pod cada 30 s, para no oscilar si la carga es intermitente. En la evidencia pasó de 6 a 2 pods en unos 2 minutos tras terminar la carga.

**¿Qué demuestra la prueba de carga?**
Un Job de dos pods con 8 hilos cada uno llamó 240 s al Service por su DNS interno: 3.858 peticiones sin ninguna fallida. Las réplicas subieron de 2 a 6 en unos 2 minutos y volvieron a 2 en unos 7 minutos totales (`docs/evidence/07_hpa.txt`).

**¿Qué pasa con el escalado manual si hay HPA?**
El HPA reajusta las réplicas según la métrica y sobrescribe `kubectl scale`. Por eso las pruebas manuales de la Fase 6 se hacen sin HPA.

