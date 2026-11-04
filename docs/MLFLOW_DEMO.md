# Guion de demostración — UI de MLflow

Objetivo: mostrar en **menos de 5 minutos** que todo el ciclo de experimentación pasa por MLflow (experimentos, comparación de runs, artefactos y Model Registry). Son 7 pasos de unos 40 segundos. Cubre REQ-MLF-07.

## Antes de empezar (2 minutos, una sola vez)

1. Arranca el servidor. La primera vez tarda varios minutos porque instala `mlflow` dentro del contenedor:

   ```bash
   docker compose up -d mlflow
   curl http://localhost:5678/health
   ```

   Debe responder `OK`.

2. Abre **http://localhost:5678** en el navegador. Si ves una pantalla de error o un `403`, el contenedor tiene el `--allowed-hosts` antiguo y no incluye `localhost:5678`; recréalo con `docker compose up -d mlflow` desde una rama con el `docker-compose.yml` actual.
3. Cierra el panel **MLflow Assistant** (la X de la derecha) y el popup **Detect Issues** ("Got it"), porque tapan la tabla.
4. La UI abre por defecto en la vista **GenAI**, que no sirve para esta demo (muestra "Failed to load chart data"). Usa las URLs directas de cada paso o el enlace **Training runs** del menú izquierdo.

Qué contiene el servidor [Cierto, medido en MLflow]:

| Elemento | Valor |
|---|---|
| Experimento | `wine-premium` (id 1), 18 runs |
| Runs del barrido con umbral ajustado por CV | 8 (tag `sweep = sweep.yaml+tuned`), nombres `NN-familia-hiperparámetros-thr` |
| Runs del barrido con umbral 0,5 | 8 (tag `sweep = sweep.yaml`) |
| Runs de reproducibilidad del baseline | 2 (`lr-baseline-run1` y `lr-baseline-run2`) |
| Modelo registrado | `wine-premium-classifier`, versiones 1, 2 y 3 |
| Aliases | `champion` → v3, `challenger` → v2 |

## Los 7 pasos

### Paso 1 — El experimento y sus runs (≈ 40 s)

- **Qué hacer:** abre
  `http://localhost:5678/#/experiments/1/runs?searchFilter=tags.sweep%20%3D%20%27sweep.yaml%2Btuned%27`
  (o en **Training runs** escribe `tags.sweep = 'sweep.yaml+tuned'` en la barra de búsqueda).
- **Qué debe verse:** "8 matching runs" con nombres legibles: `01-lr-C1.0-balanced-thr`, `02-lr-C0.1-balanced-thr`, `03-rf-n200-depthNone-balanced-thr`, `04-rf-n500-depth10-leaf2-thr`, `05-hgb-lr0.1-iter200-thr`, `06-hgb-lr0.05-iter400-leaves15-thr`, `07-svc-C1-gammaScale-thr` y `08-svc-C10-gammaScale-thr`.
- **Qué decir:** cada fila es un run con una combinación de hiperparámetros; ocho candidatos de cuatro familias, todos con los mismos splits y `SEED = 42`. Los dos runs extra del experimento (`lr-baseline-run1` y `-run2`) son la prueba de reproducibilidad: métricas idénticas.
- **Captura:** `docs/evidence/04_mlflow_01_experimento.png`

### Paso 2 — Columnas visibles (≈ 40 s)

- **Qué hacer:** botón **Columns** → marca `test_f1_premium`, `test_recall_premium`, `test_pr_auc`, `cv_f1_mean` y el parámetro `model_family`. Quita las que sobren para que la tabla quepa.
- **Qué debe verse:** la tabla con esas cinco columnas y valores en todas las filas, sin celdas vacías.
- **Qué decir:** son las métricas de `SPEC.md` §6.3. `cv_f1_mean` es la que usamos para **elegir**; `test_*` se mide una sola vez al final.
- **Captura:** `docs/evidence/04_mlflow_02_columnas.png`

### Paso 3 — Ordenar y filtrar (≈ 40 s)

- **Qué hacer:**
  1. Pulsa el encabezado de `test_f1_premium` para ordenar de mayor a menor (o usa **Sort**).
  2. En la barra de búsqueda escribe: `params.model_family = "random_forest"`.
- **Qué debe verse:** ordenado, el primero es `04-rf-n500-depth10-leaf2-thr` con `test_f1_premium` = **0,500**. Tras el filtro quedan **2 runs** (`03` y `04`).
- **Qué decir:** el orden por `test_f1_premium` no coincide exactamente con el orden por `cv_f1_mean` (por ejemplo `08-svc` es tercero en test pero último en validación cruzada), y por eso se elige con la validación cruzada y no con `test`.
- **Captura:** `docs/evidence/04_mlflow_03_orden_filtro.png`

### Paso 4 — Comparar runs (≈ 40 s)

- **Qué hacer:** quita el filtro, marca la casilla de **4 runs** (uno por familia: `04-rf-…`, `02-lr-…`, `07-svc-…`, `06-hgb-…`) y pulsa **Compare**. En la vista de comparación abre **Parallel Coordinates Plot** y elige como parámetro `decision_threshold` y como métrica `test_f1_premium`.
- **Qué debe verse:** "Comparing 4 Runs from 1 Experiment" y el gráfico de coordenadas paralelas; debajo, la tabla de parámetros y métricas lado a lado.
- **Qué decir:** el umbral de decisión cambia mucho entre modelos (0,16 a 0,62) porque se ajusta por validación cruzada; compararlos con una misma métrica y los mismos splits es lo que hace válida la comparación.
- **Captura:** `docs/evidence/04_mlflow_04_comparacion.png`

### Paso 5 — Artefactos de un run (≈ 40 s)

- **Qué hacer:** abre el run `04-rf-n500-depth10-leaf2-thr` y la pestaña **Artifacts**.
- **Qué debe verse:** `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `feature_importance.json`, `classification_report.json`, `baseline_stats.json` y el modelo registrado. Al pulsar un PNG se previsualiza.
- **Qué decir:** `baseline_stats.json` es el que alimenta el detector de drift; `feature_importance.json` muestra que `alcohol` (0,26) y `sulphates` (0,13) son las variables que más pesan.
- **Captura:** `docs/evidence/04_mlflow_05_artefactos.png`

### Paso 6 — Model Registry (≈ 40 s)

- **Qué hacer:** abre `http://localhost:5678/#/models/wine-premium-classifier` (o **Models** en el menú).
- **Qué debe verse:** la descripción del modelo y la tabla de versiones **1, 2 y 3**, cada una con su descripción, sus tags (`validated`, `decision_threshold`, `test_f1_premium`, `source_run`) y los alias: `@ challenger` en la v2 y `@ champion` en la v3.
- **Qué decir:** el servicio no carga ningún archivo: pide `models:/wine-premium-classifier@champion`, así que promover una versión es cambiar un alias, sin reconstruir la imagen. `validated = false` porque ninguna versión cumple todos los criterios de `crispdm.md` §5; lo decimos abiertamente.
- **Captura:** `docs/evidence/04_mlflow_06_registro.png`

### Paso 7 — De la versión al run (≈ 40 s)

- **Qué hacer:** abre la versión 3 (`http://localhost:5678/#/models/wine-premium-classifier/versions/3`) y pulsa el enlace de **Source Run**.
- **Qué debe verse:** `04-rf-n500-depth10-leaf2-thr` con `run_id` `1b53b6863eaa44078daf41aa54df3055`, `cv_f1_mean` = 0,568, `test_f1_premium` = 0,500, precisión 0,385 y recall 0,714.
- **Qué decir:** trazabilidad completa: del modelo servido al run, de ahí a los datos (digest de `train` y `test`, SHA-256 de los CSV en los tags) y al commit (`git_commit`, con `git_dirty` si había cambios sin commitear).
- **Captura:** `docs/evidence/04_mlflow_07_trazabilidad.png`

## Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| Página en blanco, `403` o "Request error INTERNAL_ERROR" | El contenedor tiene un `--allowed-hosts` sin `localhost:5678` | Recrear el contenedor con el `docker-compose.yml` actual: `docker compose up -d mlflow` |
| `connection reset` o `curl` sin respuesta | El contenedor sigue instalando `mlflow` (varios minutos) | Esperar y repetir `curl http://localhost:5678/health` |
| La tabla no muestra runs | Estás en la vista **GenAI** | Pulsa **Training runs** o usa las URLs de este guion |
| Faltan los modelos o las versiones | El volumen está vacío (otra máquina o `down -v`) | Restaurar la copia del volumen |
| Los artefactos no se ven | El volumen se restauró con la carpeta anidada `artifacts/artifacts` | Mover el contenido un nivel arriba (ver la restauración del volumen) |

## Ensayo cronometrado (T4.10)

1. Dev A hace la demo completa con Dev B mirando y cronometra: **meta < 5 minutos**.
2. Dev B la repite solo, sin ayuda, con este guion. Anotar el tiempo y cualquier paso donde dudó.
3. Hacer las 7 capturas durante el segundo pase y guardarlas en `docs/evidence/` con los nombres indicados.
4. La demo está lista cuando Dev B la completa solo en menos de 5 minutos.

| Pase | Quién | Tiempo | Notas |
|---|---|---|---|
| 1 | Dev A | | |
| 2 | Dev B | | |
