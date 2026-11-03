# CRISP-DM — Wine Quality MLOps

## 1. Comprensión del negocio

### Problema

Una bodega portuguesa produce vinos tintos y necesita priorizar los lotes que merecen pasar a una cata premium. El análisis fisicoquímico de laboratorio es rápido y económico, mientras que la cata de expertos es lenta y costosa. El proyecto construye un clasificador binario que estima si un lote tiene potencial premium a partir de once mediciones fisicoquímicas.

La etiqueta de negocio es `premium = 1` cuando `quality >= 7`; en los demás casos es `0`. El modelo se entrena únicamente con vinos tintos. Los vinos blancos se reservan para simular un cambio de distribución de datos y comprobar que el sistema detecta data drift antes de seguir operando como si nada hubiera cambiado.

### Stakeholders

- Enólogo jefe: interpreta las variables y valida que los resultados sean plausibles.
- Panel de cata: produce las etiquetas de calidad y comunica cambios en el criterio de evaluación.
- Gerencia comercial: define la prioridad de la línea premium y los criterios de aceptación del servicio.
- Equipo de TI/MLOps: mantiene MLflow, la API, los contenedores, Kubernetes, el monitoreo y el reentrenamiento.

### Objetivos

1. Reducir el número de lotes que deben enviarse primero a una cata exhaustiva.
2. Detectar cuándo las entradas dejan de parecerse a los tintos usados para entrenar.
3. Detectar una caída sostenida del desempeño cuando cambie el criterio de los catadores.
4. Servir predicciones reproducibles mediante una API versionada y observable.

### Criterios de éxito

Los criterios se comprobarán con el conjunto de prueba reservado y con ejecuciones registradas en MLflow:

- Negocio: `recall_premium >= 0.70`, para detectar al menos siete de cada diez lotes premium.
- Negocio: `precision_premium >= 0.50`, para que al menos la mitad de los lotes priorizados para cata premium sean realmente premium.
- Técnico: `test_f1_premium >= 0.60` y superior al baseline de regresión logística.
- Operación: el lote `monitor_red_white` y el lote `monitor_red_injected` deben producir estado ROJO, mientras `monitor_green` debe producir VERDE.
- Operación: el concept drift debe generar alarma solo después de tres lotes consecutivos bajo `F1_ref - 3σ_ref`.
- Servicio: `/health` debe responder 200 cuando el proceso está vivo, `/ready` solo cuando el modelo se cargó y `/predict` debe devolver la predicción junto con versión, `run_id` y `served_by`.

### Plan del proyecto

1. Documentar negocio, datos, preparación, modelado, evaluación y despliegue con CRISP-DM.
2. Descargar y versionar los CSV oficiales de UCI id 186 mediante SHA-256.
3. Validar, deduplicar y dividir los tintos; conservar blancos como lote de drift.
4. Entrenar modelos comparables y registrar parámetros, métricas, artefactos y modelos en MLflow.
5. Registrar versiones y aliases `champion` y `challenger`.
6. Exponer el modelo mediante FastAPI, empaquetarlo con Docker y desplegarlo en Kubernetes.
7. Implementar pruebas de data drift y concept drift, HPA, CI/CD y una UI Bootstrap.

## 6. Despliegue

### Arquitectura

El servidor MLflow conserva experimentos, artefactos y versiones del registro. La API FastAPI carga el modelo usando `MODEL_URI` y la referencia `models:/wine-premium-classifier@champion`; no carga modelos desde rutas de archivos. Docker empaqueta la API con sus dependencias. Kubernetes ejecuta tres réplicas del servicio `wine-api` en el namespace `wine-mlops`, con probes de salud y readiness. La UI Bootstrap se sirve mediante nginx y envía `/api` al Service de Kubernetes.

Flujo principal:

`MLflow → API FastAPI → imagen Docker → Deployment/Service de Kubernetes → UI`

### Monitoreo

Cada lote nuevo se compara con las distribuciones del baseline de entrenamiento mediante KS y PSI para las once variables de entrada. También se compara la distribución de la clase predicha con chi-cuadrado. El lote es ROJO si cumple la regla combinada definida en `SPEC.md` §7.1; la CLI devuelve código 1 para que CI pueda bloquear una promoción. Los reportes JSON se guardan en `reports/`.

El concept drift se simula reetiquetando lotes de 200 observaciones, sin modificar sus entradas. Se calcula F1 de la clase premium por lote con el modelo `champion`. La alarma requiere tres lotes consecutivos por debajo de `F1_ref - 3σ_ref`.

### Reentrenamiento y promoción

Cuando la alarma persiste, se entrena un candidato con datos recientes y etiquetas disponibles. El candidato se registra como `challenger` y solo se promueve a `champion` si supera al modelo actual en el lote de validación definido. La API recibe el nuevo alias al reiniciarse y expone la versión y el `run_id` servidos.

### Retraso de etiquetas

Las etiquetas de cata pueden llegar después de la predicción. Mientras están pendientes, se monitorean las entradas, la clase predicha y la distribución de probabilidades. De cada lote se solicita una muestra etiquetada del 5–10 % para obtener una estimación temprana de F1. Cuando llegan las etiquetas completas, se actualiza el cálculo de concept drift y se decide si corresponde reentrenar.
