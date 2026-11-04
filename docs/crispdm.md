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

- Negocio: `test_recall_premium >= 0.70`, para detectar al menos siete de cada diez lotes premium.
- Negocio: `test_precision_premium >= 0.50`, para que al menos la mitad de los lotes priorizados para cata premium sean realmente premium.
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

Se usa Wine Quality (Cortez et al., 2009), del UCI Machine Learning Repository, dataset id 186 (https://archive.ics.uci.edu/dataset/186/wine+quality, DOI 10.24432/C56S3T, confirmado en la página oficial). Son vinos *vinho verde* del norte de Portugal. La versión es la oficial y queda fijada por el SHA-256 de cada CSV, que se registra en `data/DATA_CARD.md` y como tag en cada run de MLflow. No se usan `sklearn.datasets.load_wine()` (178 registros, otro problema) ni variantes de Kaggle como WineQT (subconjunto sin trazabilidad).

| Archivo | Filas | Uso |
|---|---|---|
| `winequality-red.csv` | 1.599 | Entrenamiento, prueba y lote de monitoreo del mismo origen |
| `winequality-white.csv` | 4.898 | Solo lote de data drift (`monitor_red_white`) |

El total es 6.497 registros, por encima del mínimo de 1.000 que exige la consigna. El subconjunto de tintos con el que se entrena (1.599 filas, 1.359 tras deduplicar) también lo supera. El separador es `;`.

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

- **Duplicados [Cierto, medido con el ETL]:** 240 filas exactas repetidas en los tintos (1.599 → 1.359) y 937 en los blancos (4.898 → 3.961). Ninguna fila comparte las once entradas con una etiqueta distinta.
- **Desbalance [Cierto]:** el 13,5 % de los tintos deduplicados tiene `quality >= 7` (13,6 % antes de limpiar). En los blancos es el 20,8 %. Por eso la métrica principal no es la exactitud (D-04).
- **Valores faltantes [Cierto]:** ninguno en los dos archivos. El ETL lo valida de todos modos, porque un lote futuro sí podría traerlos.
- **Rangos [Cierto]:** la densidad de los blancos llega a 1,03898 g/cm³, fuera del rango de los tintos (máximo 1,00369). Es la señal más visible de que son otra población.

### Sesgos y límites

- Todos los vinos son de una sola región (Vinho Verde); el modelo no debe extrapolarse a otras denominaciones.
- La etiqueta depende del criterio subjetivo de los catadores. Ese criterio puede cambiar con el tiempo, lo que motiva el monitoreo de concept drift.
- Un modelo entrenado con tintos no es válido para blancos: los rangos de `residual_sugar`, `total_sulfur_dioxide` y `free_sulfur_dioxide` son distintos. Este contraste es el que se aprovecha para probar la detección de data drift.
- Con pocos positivos en el conjunto de prueba (28 vinos premium entre 204), las métricas son ruidosas; se reporta también el promedio y la desviación de la validación cruzada.

## 3. Preparación de los datos

1. **Selección:** se entrena únicamente con los tintos (D-02). Los blancos se reservan para el lote de drift.
2. **Normalización de nombres:** de los nombres del CSV a `snake_case`, con el mapeo declarado en `src/wine/config.py`.
3. **Validación en dos niveles:** el ETL exige las columnas exactas, tipo numérico y ausencia de nulos, y aplica dos tipos de rango. Los **límites de integridad** (valores físicamente imposibles, por ejemplo `ph` fuera de 0–14) se aplican siempre. Los **rangos de tintos** (`ph` 2,5–4,5, `alcohol` 8–15, `density` 0,98–1,01) solo se aplican al entrenamiento: un lote de blancos con densidad de hasta 1,039 debe llegar al detector de drift y ser marcado ROJO, no abortar el pipeline. Si algo falla, el error nombra la columna.
4. **Limpieza:** se eliminan los duplicados exactos **antes** de dividir (D-03): 240 en los tintos y 937 en los blancos. Si se dividiera primero, filas idénticas podrían caer en `train` y `test` y inflar las métricas. Los blancos se deduplican antes de muestrear las 500 filas.
5. **Objetivo derivado:** `premium = quality >= 7`.
6. **Particiones**, todas con `SEED = 42` y guardadas en `data/processed/` con su SHA-256:

| Conjunto | Origen | Proporción o tamaño | Para qué sirve |
|---|---|---|---|
| `train` | Tintos deduplicados | 951 filas (70 %, 13,6 % premium) | Entrenamiento y validación cruzada de 5 folds |
| `test` | Tintos deduplicados | 204 filas (15 %, 13,7 % premium) | Evaluación final y base de la simulación de concept drift |
| `monitor_green` | Tintos deduplicados | 204 filas (15 %, 13,2 % premium) | Lote del mismo origen: debe dar VERDE |
| `monitor_red_white` | Blancos deduplicados | 500 filas con semilla fija (20,4 % premium) | Lote derivado real: debe dar ROJO |
| `monitor_red_injected` | `monitor_green` con `alcohol` +1,5, `residual_sugar` ×3 y `total_sulfur_dioxide` ×2 | 204 filas | Lote derivado inyectado: debe dar ROJO |

La estratificación por `premium` conserva la proporción de la clase minoritaria en cada partición (13,2 %–13,7 %). Los splits se verifican sin fila repartida entre conjuntos y con los mismos SHA-256 en dos ejecuciones. Tener un conjunto de monitoreo propio del mismo origen permite medir los falsos ROJOS sin tocar `test`.

## 4. Modelado

### Candidatos

Solo scikit-learn (D-05), con el preprocesamiento dentro del `Pipeline` para que el modelo registrado reciba las columnas crudas:

| Familia | Preprocesamiento | Motivo |
|---|---|---|
| Regresión logística (baseline) | `StandardScaler` | Referencia simple e interpretable; todo modelo debe superarla |
| Random Forest | — | Captura interacciones no lineales y es robusto al ruido |
| HistGradientBoosting | — | Boosting eficiente en datos tabulares pequeños |
| SVC con kernel RBF | `StandardScaler` | Frontera no lineal; necesita `probability=True` para obtener la probabilidad |

Se ejecutan ocho combinaciones de hiperparámetros (dos por familia), definidas en `experiments/sweep.yaml`, todas con los mismos splits y la misma semilla. El barrido se ejecuta dos veces: con el umbral fijo de 0,5 y con el umbral ajustado por validación cruzada; la selección usa solo la segunda, para comparar cada candidato en su mejor umbral.

### Validación

- Validación cruzada estratificada de 5 folds sobre `train`; se reportan `cv_f1_mean` y `cv_f1_std`.
- La familia, los hiperparámetros y el umbral de decisión se eligen únicamente con la validación cruzada sobre `train`: `cv_f1_mean` y las predicciones fuera de fold. El umbral se elige sobre una malla de 0,05 a 0,95 en pasos de 0,01, como el que maximiza el F1 premium de las predicciones fuera de fold, y se registra como parámetro del run (`decision_threshold`, con `threshold_source = cv_oof`).
- `test` se evalúa una sola vez por candidato, ya con el umbral fijado, y no interviene en ninguna elección. Así los criterios de la sección 1 se miden sobre datos no vistos, y el `F1_ref` del concept drift (calculado sobre `test`) no queda inflado.
- Cada ejecución queda en MLflow con sus parámetros, métricas, artefactos y la firma del modelo. No se guarda ningún modelo con `pickle` ni `joblib`.

### Métrica principal

La métrica principal es `test_f1_premium`, el F1 de la clase premium (D-04). Con 13,5 % de positivos, un modelo que siempre responda "no premium" tendría 86 % de exactitud y F1 igual a cero. El F1 resume el equilibrio entre precisión y recall de la clase que le importa a la bodega. Se informan además `test_precision_premium`, `test_recall_premium`, `test_roc_auc`, `test_pr_auc` y `test_balanced_accuracy`. Los criterios de la sección 1 usan F1, precisión y recall de la clase premium; ROC-AUC, PR-AUC y la exactitud balanceada son métricas de diagnóstico.

## 5. Evaluación

### Criterios de aceptación

Son los de la sección 1, medidos sobre `test` una sola vez por candidato. Los tres umbrales numéricos son una **[Decisión]** del equipo y están pendientes de reflejarse en `SPEC.md`:

| Tipo | Métrica | Criterio |
|---|---|---|
| Negocio | `test_recall_premium` | ≥ 0,70 |
| Negocio | `test_precision_premium` | ≥ 0,50 |
| Técnico | `test_f1_premium` | ≥ 0,60 y superior al baseline de regresión logística |
| Técnico | `cv_f1_mean ± cv_f1_std` | `test_f1_premium` dentro de `cv_f1_mean ± 2·cv_f1_std`; si no, se documenta la causa |

### Resultados del barrido [Cierto]

Ocho candidatos con el umbral ajustado por validación cruzada, ordenados por `cv_f1_mean`. La línea de base (regresión logística, umbral 0,5) tiene `cv_f1_mean` 0,514 ± 0,046.

| Run | Umbral | `cv_f1_mean` ± std | `test_f1` | Precisión | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| **RF n=500, prof. 10, hoja 2** | **0,22** | **0,568 ± 0,040** | **0,500** | **0,385** | **0,714** | **0,496** |
| LR C=0,1 balanced | 0,62 | 0,542 ± 0,059 | 0,463 | 0,352 | 0,679 | 0,451 |
| RF n=200 balanced | 0,35 | 0,540 ± 0,042 | 0,482 | 0,364 | 0,714 | 0,535 |
| LR C=1 balanced | 0,62 | 0,538 ± 0,060 | 0,463 | 0,352 | 0,679 | 0,456 |
| SVC C=1 | 0,16 | 0,537 ± 0,031 | 0,431 | 0,378 | 0,500 | 0,488 |
| HGB lr=0,05, 400 iter | 0,18 | 0,519 ± 0,068 | 0,386 | 0,379 | 0,393 | 0,506 |
| HGB lr=0,1, 200 iter | 0,10 | 0,517 ± 0,050 | 0,414 | 0,400 | 0,429 | 0,506 |
| SVC C=10 | 0,17 | 0,486 ± 0,038 | 0,478 | 0,410 | 0,571 | 0,516 |

Las diferencias entre los cinco primeros caben dentro de una desviación de la validación cruzada: el ranking no es concluyente. Con solo 28 positivos en `test`, una diferencia de 0,03 en F1 equivale a uno o dos vinos.

### Modelo seleccionado frente a los criterios [Cierto]

El Random Forest (n=500, profundidad 10, umbral 0,22) es el de mayor `cv_f1_mean`. Es la versión 3 y el `champion`. Los factores que más pesan en sus predicciones (importancia por permutación) son `alcohol` (0,26), `sulphates` (0,13), `total_sulfur_dioxide` (0,09) y `volatile_acidity` (0,08).

| Criterio | Valor obtenido | ¿Cumple? |
|---|---|---|
| `test_recall_premium` ≥ 0,70 | 0,714 | Sí |
| `test_precision_premium` ≥ 0,50 | 0,385 | **No** |
| `test_f1_premium` ≥ 0,60 | 0,500 | **No** |
| `test_f1_premium` superior al baseline (0,469) | 0,500 | Sí |
| `test_f1_premium` dentro de `cv_f1_mean ± 2·cv_f1_std` (0,488–0,648) | 0,500 | Sí |

**El modelo no alcanza los criterios de precisión y de F1.** Lo que se observa: con 13,5 % de positivos y un recall de 0,71, el modelo marca como premium alrededor de uno de cada cuatro vinos (52 de 204) y solo 20 de esos 52 lo son. Ajustar el umbral con las predicciones fuera de fold no cierra la brecha; no se probó si más datos, más variables u otras técnicas la cerrarían. Por la regla siguiente, el `champion` queda con `validated=false`; la brecha se documenta como limitación y no se ocultó.

### Regla para promover un modelo

Un modelo es candidato a `champion` solo si cumple los tres criterios numéricos de la tabla. Entre los que los cumplen, se elige el de mayor `cv_f1_mean`; si dos empatan dentro de un `cv_f1_std`, se prefiere el más simple. Si ninguno cumple, se ajusta el umbral de decisión usando solo las predicciones fuera de fold de `train` (registrado como parámetro del run) y, si la brecha persiste, el `champion` es el de mayor `cv_f1_mean` con el tag `validated=false` y la brecha se documenta como limitación sin ocultar el resultado.

### Versiones del registro

Se registran tres versiones de `wine-premium-classifier`, todas con descripción, `run_id` y el tag `validated=false` (ninguna cumple los tres criterios):

| Versión | Run | Umbral | Alias |
|---|---|---|---|
| v1 | Regresión logística C=1 (baseline) | 0,50 | — |
| v2 | Regresión logística C=0,1 | 0,62 | `challenger` |
| v3 | Random Forest n=500 | 0,22 | `champion` |

Como el mejor run global es un Random Forest, v2 es el mejor run de otra familia, para que las tres versiones sean runs distintos. El modelo devuelve `[P(no premium), P(premium)]`; el umbral de decisión vive en el tag `decision_threshold` de cada versión.

### Evaluación después del despliegue

El desempeño no se da por bueno para siempre. El `champion` se evalúa de nuevo con el monitoreo de la sección 6: data drift sobre las entradas y concept drift sobre el F1 de la clase premium. Resultados medidos (detalle en `docs/DRIFT.md`):

- **Data drift:** `monitor_green` da VERDE (0 de 11 variables); `monitor_red_white` da ROJO (10 de 11 variables) y `monitor_red_injected` da ROJO (3 de 11, más el chi² de la clase predicha). La regla `k = 3` produce 1 falso ROJO en 200 lotes bootstrap (0,5 %).
- **Concept drift:** con la regla de reetiquetado de la SPEC el F1 por lote se mantiene entre 0,43 y 0,64, sobre el umbral de 0,282, y **no salta la alarma**. El modelo depende sobre todo de `alcohol` y `sulphates`, y la regla nueva marca premium a un subconjunto de lo que ya marca, por lo que no lo degrada. El mecanismo de alarma sí se verifica con un modelo oráculo (`docs/DRIFT.md` §2.4.1). La comparación entre `champion` y `challenger` usa siempre la misma métrica principal y el mismo lote: el último lote etiquetado, que no puede formar parte de los datos de entrenamiento del `challenger`.

## 6. Despliegue

### Arquitectura

El servidor MLflow conserva experimentos, artefactos y versiones del registro. La API FastAPI carga el modelo usando `MODEL_URI` y la referencia `models:/wine-premium-classifier@champion`; no carga modelos desde rutas de archivos. Docker empaqueta la API con sus dependencias. Kubernetes ejecuta dos réplicas del servicio `wine-api` en el namespace `wine-mlops`, con probes de salud y readiness. La UI Bootstrap se sirve mediante nginx y envía `/api` al Service de Kubernetes.

Flujo principal:

`MLflow → API FastAPI → imagen Docker → Deployment/Service de Kubernetes → UI`

### Monitoreo

Cada lote nuevo se compara con las distribuciones del baseline de entrenamiento mediante KS y PSI para las once variables de entrada. También se compara la distribución de la clase predicha con chi-cuadrado. El lote es ROJO si cumple la regla combinada definida en `SPEC.md` §7.1; la CLI devuelve código 1 para que CI pueda bloquear una promoción. Los reportes JSON se guardan en `reports/`.

El concept drift se simula reetiquetando lotes de 200 observaciones, sin modificar sus entradas. Se calcula F1 de la clase premium por lote con el modelo `champion`. La alarma requiere tres lotes consecutivos por debajo de `F1_ref - 3σ_ref`.

### Reentrenamiento y promoción

Cuando la alarma persiste, se entrena un candidato con datos recientes y etiquetas disponibles. El candidato se registra como `challenger` y solo se promueve a `champion` si supera al modelo actual en el último lote etiquetado, que no se usó para entrenarlo. La API recibe el nuevo alias al reiniciarse y expone la versión y el `run_id` servidos.

### Retraso de etiquetas

Las etiquetas de cata pueden llegar después de la predicción. Mientras están pendientes, se monitorean las entradas, la clase predicha y la distribución de probabilidades. De cada lote se solicita una muestra etiquetada del 5–10 % para obtener una estimación temprana de F1. Cuando llegan las etiquetas completas, se actualiza el cálculo de concept drift y se decide si corresponde reentrenar.
