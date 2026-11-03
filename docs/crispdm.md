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

## 2. Comprensión de los datos

### Fuente y versión

Se usa Wine Quality (Cortez et al., 2009), del UCI Machine Learning Repository, dataset id 186 (https://archive.ics.uci.edu/dataset/186/wine+quality, DOI 10.24432/C56S3T). Son vinos *vinho verde* del norte de Portugal. La versión es la oficial y queda fijada por el SHA-256 de cada CSV, que se registra en `data/DATA_CARD.md` y como tag en cada run de MLflow. No se usan `sklearn.datasets.load_wine()` (178 registros, otro problema) ni variantes de Kaggle como WineQT (subconjunto sin trazabilidad).

| Archivo | Filas | Uso |
|---|---|---|
| `winequality-red.csv` | 1.599 | Entrenamiento, prueba y lote de monitoreo del mismo origen |
| `winequality-white.csv` | 4.898 | Solo lote de data drift (`monitor_red_white`) |

El total es 6.497 registros, por encima del mínimo de 1.000 que exige la consigna. El separador es `;`.

### Variables

Las once variables de entrada son numéricas continuas, resultado del análisis fisicoquímico de laboratorio. En el código se normalizan a `snake_case`; la columna `wine_type` se agrega solo para análisis y nunca es entrada del modelo.

| Variable | Unidad |
|---|---|
| `fixed_acidity` | g/dm³ (ácido tartárico) |
| `volatile_acidity` | g/dm³ (ácido acético) |
| `citric_acid` | g/dm³ |
| `residual_sugar` | g/dm³ |
| `chlorides` | g/dm³ (cloruro de sodio) |
| `free_sulfur_dioxide` | mg/dm³ |
| `total_sulfur_dioxide` | mg/dm³ |
| `density` | g/cm³ |
| `ph` | — |
| `sulphates` | g/dm³ (sulfato de potasio) |
| `alcohol` | % vol |

El objetivo original es `quality`, un entero asignado por la mediana de al menos tres catadores. A partir de él se deriva `premium = 1` si `quality >= 7`, en otro caso `0` (D-01).

### Calidad esperada

- **Duplicados:** la literatura reporta cerca de 240 filas exactas repetidas en los tintos **[Verificar]**. La cifra real se obtendrá con el ETL (Fase 3) y reemplazará este texto.
- **Desbalance:** se espera entre 13 % y 14 % de tintos con `quality >= 7` **[Verificar]**. Por eso la métrica principal no es la exactitud (D-04).
- **Valores faltantes:** la versión oficial no tiene nulos. El ETL lo valida de todos modos, porque un lote futuro sí podría traerlos.

### Sesgos y límites

- Todos los vinos son de una sola región (Vinho Verde); el modelo no debe extrapolarse a otras denominaciones.
- La etiqueta depende del criterio subjetivo de los catadores. Ese criterio puede cambiar con el tiempo, lo que motiva el monitoreo de concept drift.
- Un modelo entrenado con tintos no es válido para blancos: los rangos de `residual_sugar`, `total_sulfur_dioxide` y `free_sulfur_dioxide` son distintos. Este contraste es el que se aprovecha para probar la detección de data drift.
- Con pocos positivos en el conjunto de prueba (aproximadamente 28 lotes premium), las métricas son ruidosas; se reporta también el promedio y la desviación de la validación cruzada.

## 3. Preparación de los datos

1. **Selección:** se entrena únicamente con los tintos (D-02). Los blancos se reservan para el lote de drift.
2. **Normalización de nombres:** de los nombres del CSV a `snake_case`, con el mapeo declarado en `src/wine/config.py`.
3. **Validación:** el ETL exige las columnas exactas, tipo `float`, ausencia de nulos y rangos físicos plausibles (por ejemplo `ph` entre 2,5 y 4,5, `alcohol` entre 8 y 15, `density` entre 0,98 y 1,01). Si algo falla, el pipeline termina con un error que nombra la columna.
4. **Limpieza:** se eliminan los duplicados exactos **antes** de dividir (D-03). Si se dividiera primero, filas idénticas podrían caer en `train` y `test` y inflar las métricas.
5. **Objetivo derivado:** `premium = quality >= 7`.
6. **Particiones**, todas con `SEED = 42` y guardadas en `data/processed/` con su SHA-256:

| Conjunto | Origen | Proporción o tamaño | Para qué sirve |
|---|---|---|---|
| `train` | Tintos deduplicados | 70 % estratificado | Entrenamiento y validación cruzada de 5 folds |
| `test` | Tintos deduplicados | 15 % estratificado | Evaluación final y base de la simulación de concept drift |
| `monitor_green` | Tintos deduplicados | 15 % estratificado | Lote del mismo origen: debe dar VERDE |
| `monitor_red_white` | Blancos | 500 filas con semilla fija | Lote derivado real: debe dar ROJO |
| `monitor_red_injected` | `monitor_green` con desplazamientos | Igual que `monitor_green` | Lote derivado inyectado: debe dar ROJO |

La estratificación por `premium` conserva la proporción de la clase minoritaria en cada partición. Tener un conjunto de monitoreo propio del mismo origen permite medir los falsos ROJOS sin tocar `test`.

## 4. Modelado

### Candidatos

Solo scikit-learn (D-05), con el preprocesamiento dentro del `Pipeline` para que el modelo registrado reciba las columnas crudas:

| Familia | Preprocesamiento | Motivo |
|---|---|---|
| Regresión logística (baseline) | `StandardScaler` | Referencia simple e interpretable; todo modelo debe superarla |
| Random Forest | — | Captura interacciones no lineales y es robusto al ruido |
| HistGradientBoosting | — | Boosting eficiente en datos tabulares pequeños |
| SVC con kernel RBF | `StandardScaler` | Frontera no lineal; necesita `probability=True` para obtener la probabilidad |

Se ejecutan ocho combinaciones de hiperparámetros (dos por familia), definidas en `experiments/sweep.yaml`, todas con los mismos splits y la misma semilla.

### Validación

- Validación cruzada estratificada de 5 folds sobre `train`; se reportan `cv_f1_mean` y `cv_f1_std`.
- Evaluación final sobre `test`, que no se usa para elegir hiperparámetros salvo para decidir entre los ocho candidatos ya entrenados y el umbral de decisión, que se registra como parámetro.
- Cada ejecución queda en MLflow con sus parámetros, métricas, artefactos y la firma del modelo. No se guarda ningún modelo con `pickle` ni `joblib`.

### Métrica principal

La métrica principal es `test_f1_premium`, el F1 de la clase premium (D-04). Con aproximadamente 14 % de positivos, un modelo que siempre responda "no premium" tendría 86 % de exactitud y F1 igual a cero. El F1 resume el equilibrio entre precisión y recall de la clase que le importa a la bodega. Se informan además `test_precision_premium`, `test_recall_premium`, `test_roc_auc`, `test_pr_auc` y `test_balanced_accuracy`, que son las mismas métricas que usan los criterios de la sección 1.

## 5. Evaluación

### Criterios de aceptación

Son los de la sección 1, medidos sobre `test`:

| Tipo | Métrica | Criterio |
|---|---|---|
| Negocio | `test_recall_premium` | ≥ 0,70 |
| Negocio | `test_precision_premium` | ≥ 0,50 |
| Técnico | `test_f1_premium` | ≥ 0,60 y superior al baseline de regresión logística |
| Técnico | `cv_f1_mean ± cv_f1_std` | Informado; el resultado de `test` debe caer dentro de un rango razonable de la validación cruzada |

### Regla para promover un modelo

Un modelo es candidato a `champion` solo si cumple los tres criterios numéricos de la tabla. Entre los que los cumplen, se elige el de mayor `test_f1_premium`; si dos empatan dentro de una desviación de la validación cruzada, se prefiere el más simple. Si ninguno cumple, se ajusta el umbral de decisión (registrado como parámetro del run) y, si la brecha persiste, se documenta como limitación sin ocultar el resultado.

### Versiones del registro

Se registran como mínimo tres versiones de `wine-premium-classifier`: v1 (baseline), v2 (mejor Random Forest, alias `challenger`) y v3 (mejor run global, alias `champion`). Cada versión lleva descripción, `run_id` y el tag `validated`.

### Evaluación después del despliegue

El desempeño no se da por bueno para siempre. El `champion` se evalúa de nuevo con el monitoreo de la sección 6: data drift sobre las entradas y concept drift sobre el F1 de la clase premium. La comparación entre `champion` y `challenger` usa siempre la misma métrica principal y el mismo lote de validación.

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
