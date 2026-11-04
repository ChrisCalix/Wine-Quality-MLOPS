# Diseño de drift — Wine Quality MLOps

Este documento define **qué se mide, con qué prueba, con qué umbral y por qué**. La implementación llega en la Fase 3 (`src/wine/drift/`). Los umbrales viven en `SPEC.md` §7 y en `src/wine/config.py`; si cambian, cambian en los tres sitios y con PR aprobado por ambos.

Hay dos fenómenos distintos y se simulan por separado:

| Fenómeno | Qué cambia | Cómo se simula | Qué se vigila |
|---|---|---|---|
| Data drift | La distribución de las **entradas** | Vinos blancos y un lote tinto desplazado | Las 11 variables y la clase predicha |
| Concept drift | La relación **entrada → etiqueta** | Reetiquetado de lotes tintos, sin tocar las entradas | El F1 de la clase premium por lote |

Los blancos **no** sirven para probar concept drift, porque además de la etiqueta cambian las entradas.

---

## 1. Data drift

### 1.1 Baseline y lotes

- **Baseline:** el conjunto `train` (tintos deduplicados, 70 %). Sus estadísticos se guardan en `baseline_stats.json` en cada run de MLflow.
- **Lote:** cualquier CSV con las 11 variables. Se compara contra el baseline variable por variable.

| Escenario | Origen | Resultado esperado |
|---|---|---|
| `monitor_green` | Tintos deduplicados, 15 % estratificado | **VERDE** (mismo origen) |
| `monitor_red_white` | 500 blancos con `SEED = 42` | **ROJO** (derivada real) |
| `monitor_red_injected` | `monitor_green` con desplazamientos sintéticos | **ROJO** (derivada inyectada) |

Desplazamientos de `monitor_red_injected`:

| Variable | Desplazamiento |
|---|---|
| `alcohol` | `+1,5` (% vol) |
| `residual_sugar` | `× 3` |
| `total_sulfur_dioxide` | `× 2` |

Son tres variables, el mínimo que exige la regla ROJO de §1.3. Para `monitor_red_white` se espera ROJO por `residual_sugar`, `total_sulfur_dioxide` y `free_sulfur_dioxide`, entre otras **[Verificar en la Fase 3]**.

### 1.2 Pruebas

Las once entradas son numéricas continuas, así que hay un solo tipo de variable de entrada. Para cubrir también una variable categórica se monitorea la **salida del modelo** (clase predicha, `premium` 0/1).

| Prueba | Se aplica a | Por qué esta prueba | Umbral de drift | Origen del número |
|---|---|---|---|---|
| Kolmogorov–Smirnov de dos muestras | Cada una de las 11 variables | No supone normalidad y compara la distribución completa de una variable continua | p-valor < 0,05 / 11 ≈ **0,0045** | α = 0,05 con corrección de Bonferroni por 11 comparaciones, para controlar las falsas alarmas |
| PSI (*Population Stability Index*) | Cada una de las 11 variables | Mide la **magnitud** del cambio. Con muestras grandes el KS detecta cambios irrelevantes; el PSI no | < 0,10 estable · 0,10–0,25 vigilar · **≥ 0,25 drift** | Convención del riesgo crediticio para estabilidad poblacional |
| Chi-cuadrado | Distribución de la clase predicha | Prueba estándar para variables categóricas; sirve como señal cuando aún no hay etiquetas | p-valor < **0,01** | Más estricto que 0,05 porque es una sola prueba y con mucho peso en la regla |

Detalles de cálculo:

- **PSI:** 10 intervalos por cuantiles del baseline, con los extremos abiertos para que un valor fuera del rango de entrenamiento caiga en el primer o el último intervalo en lugar de perderse. Los intervalos vacíos se suavizan con un epsilon pequeño para evitar logaritmos de cero.
- **KS:** `scipy.stats.ks_2samp` entre el baseline y el lote.
- **Chi²:** se compara, con una tabla de contingencia 2×2, la proporción de `premium = 1` predicha en el lote con la predicha sobre el baseline (`train`). Las predicciones salen del `champion`, aplicando el umbral de su versión. Es opcional en la CLI: sin modelo se omite y el reporte lo anota; con `MLFLOW_TRACKING_URI` definido o `--model-uri` se incluye.
- La **corrección de Bonferroni** y el **PSI** se combinan a propósito: el KS solo dice que "hay diferencia" y el PSI dice si esa diferencia es grande. Una variable cuenta como con drift solo si cumple **ambas** condiciones.

### 1.3 Regla del lote

| Estado | Condición |
|---|---|
| **ROJO** | **≥ k de 11** variables con drift (KS significativo **y** PSI ≥ 0,25), **o** chi² de la clase predicha significativo |
| **AMARILLO** | 1 o 2 variables con drift, o alguna variable con PSI entre 0,10 y 0,25 |
| **VERDE** | En cualquier otro caso |

- **Valor inicial: k = 3.** Es una **[Decisión]** provisional. Se calibra en la Fase 3 (T3.14): se generan 200 lotes bootstrap de `monitor_green` y la regla debe dar **falsos ROJOS en menos del 5 %** de ellos. Si no se cumple, se ajusta `k` y el valor final y la tabla de calibración se anotan en la sección 3 de este documento.
- La CLI `python -m wine.drift.data --batch <csv> --out reports/<nombre>.json` termina con código `1` en ROJO y `0` en VERDE o AMARILLO, para que CI pueda bloquear una promoción.
- El reporte JSON trae una entrada por cada una de las 11 variables, con su estadístico KS, su p-valor, su PSI y si cuenta como drift, además del resultado del chi² y el estado del lote.

### 1.4 Criterios de aceptación

- `monitor_green` → VERDE.
- `monitor_red_white` → ROJO.
- `monitor_red_injected` → ROJO.
- Falsos ROJOS < 5 % en 200 lotes bootstrap de `monitor_green`.

Las pruebas automáticas están en `tests/test_data_drift.py` (Fase 3).

---

## 2. Concept drift

### 2.1 Qué se simula

El **nuevo panel de catadores** cambia su criterio: solo considera premium a un vino con `alcohol >= 11.5` **y** `sulphates >= 0.65`. Las entradas de los lotes no cambian; solo cambia la etiqueta de algunas filas. Es la situación que ocurre cuando cambia el panel de cata o la normativa.

### 2.2 Regla de reetiquetado

- Se generan **12 lotes bootstrap de 200 filas** a partir de `test`, con `SEED = 42`.
- Los lotes 1 a 4 conservan las etiquetas originales.
- Desde el **lote 5**, una fracción creciente de las filas del lote se reetiqueta con la regla nueva: **10 %** en el lote 5, **20 %** en el 6, y así sucesivamente hasta **70 %** en el lote 11. El lote 12 mantiene el 70 %. Las filas reetiquetadas se eligen al azar con la semilla fija. **[Decisión]** El reparto exacto de fracciones por lote se fija así para que el drift sea gradual.
- Las filas no seleccionadas conservan su etiqueta original.

### 2.3 Métrica y umbral

- **Métrica por lote:** `f1_premium`, calculada con el modelo `champion` sobre las entradas del lote y las etiquetas (reetiquetadas o no).
- **Referencia:** `F1_ref` y `σ_ref` son la media y la desviación estándar del F1 en **200 lotes bootstrap de `test` sin reetiquetar**.
- **Umbral de caída:** un lote está en caída si su F1 < `F1_ref − 3 · σ_ref`.
- **Origen del número:** el umbral sale de la variabilidad natural del propio modelo, no de una cifra arbitraria. Con un `test` de unos 28 positivos el F1 es ruidoso, y por eso se mide su dispersión antes de decidir qué es una caída real.

### 2.4 Persistencia y alarma

- La alarma se dispara con **3 lotes consecutivos** en caída. Un lote ruidoso aislado no la activa.
- Pruebas automáticas (`tests/test_concept_drift.py`, 5 semillas): sin reetiquetado no hay alarma; la persistencia exige 3 lotes consecutivos; y el mecanismo sí alarma cuando el modelo se degrada de verdad (se comprueba con un modelo oráculo que acierta las etiquetas originales).
- Salidas: `reports/concept_drift.json` y `reports/concept_drift.png` (F1 por lote, línea de umbral, inicio del drift y lote de alarma marcados).

### 2.4.1 Resultado con el `champion` real: no hay alarma [Cierto]

Con la regla de la sección 2.2 y el `champion` (v3, umbral 0,22), `SEED = 42`: `F1_ref` = 0,495, `σ_ref` = 0,071 y umbral = 0,282. El F1 por lote es 0,52 · 0,43 · 0,64 · 0,62 · 0,57 · 0,59 · 0,49 · 0,49 · 0,49 · 0,44 · 0,53 · 0,54. **Ningún lote cae bajo el umbral y no se dispara la alarma**, de modo que `reports/concept_drift.png` no marca ningún lote de alarma.

Esto no es un fallo del código sino una propiedad del escenario con este modelo:

- El modelo depende sobre todo de `alcohol` y `sulphates`, y marca premium a los vinos con valores altos de ambos. La regla nueva exige `alcohol >= 11.5` y `sulphates >= 0.65`, así que marca premium a un subconjunto de los vinos que el modelo ya marca.
- Sobre `test`, la regla marca 20 vinos premium frente a 28 originales, y el modelo predice 52. Si se reetiqueta el 100 %, el **recall sube** de 0,71 a 0,95, la precisión baja de 0,38 a 0,37 y el F1 pasa de 0,500 a 0,528.
- Es decir, este cambio de criterio no es drift para este modelo: no empeora sus aciertos.

**Consecuencia para los requisitos.** REQ-CD-01 (simular el cambio) y REQ-CD-03 (criterio de reentrenamiento) están definidos e implementados. REQ-CD-02 pide la gráfica con el lote de alarma, que con este escenario queda sin alarma. La demostración de que la alarma funciona se apoya en la prueba con modelo oráculo, no en el `champion`. Se deja como decisión abierta del equipo si se quiere definir otro escenario de reetiquetado que sí degrade al modelo.

### 2.5 Acción de reentrenamiento

1. La alarma dispara el reentrenamiento de un candidato con datos recientes que ya tienen etiqueta.
2. El candidato se registra en MLflow como `challenger`.
3. Se evalúa contra el `champion` en el **último lote etiquetado**, que no puede haber formado parte del entrenamiento del candidato.
4. Solo se promueve a `champion` si lo supera en ese lote con la misma métrica principal. La API recibe el nuevo alias al reiniciarse y expone la versión y el `run_id` servidos.

---

## 3. Calibración de la regla "k de 11"

Resultado de `python -m wine.drift.calibrate` (T3.14): 200 lotes bootstrap de `monitor_green` (con reemplazo, mismo tamaño), `SEED = 42`, regla completa con el chi² del `champion` (v3, umbral 0,22). Datos en `reports/calibration_data_drift.json`.

| k | Falsos ROJOS (de 200) | Tasa | ¿Cumple < 5 %? | ¿`monitor_red_white` sigue ROJO? | ¿`monitor_red_injected` sigue ROJO? |
|---|---|---|---|---|---|
| 1 | 4 | 2,0 % | Sí | Sí | Sí |
| 2 | 1 | 0,5 % | Sí | Sí | Sí |
| **3** | **1** | **0,5 %** | **Sí** | **Sí** | **Sí** |
| 4 | 1 | 0,5 % | Sí | Sí | **No** |
| 5 | 1 | 0,5 % | Sí | Sí | **No** |
| 6 | 1 | 0,5 % | Sí | Sí | **No** |

**Valor final: `k = 3`.** Es el mayor `k` que cumple el criterio y sigue detectando el lote inyectado (que desplaza exactamente tres variables). Con `k = 1` o `2` desaparecería la zona AMARILLA, y con `k ≥ 4` el lote inyectado dejaría de ser ROJO. No hace falta cambiar `DRIFT_VARIABLES_FOR_RED` en `config.py`.

Sin el chi² (solo KS y PSI) los falsos ROJOS con `k = 3` son 0 de 200. El único falso ROJO de la tabla lo produce el chi².

**Dos límites que conviene conocer**

- **La referencia del chi² importa.** Con `test` (204 filas, tasa predicha 25,5 %) como referencia, los falsos ROJOS con `k = 3` suben a 21 de 200 (10,5 %), porque `monitor_green` (18,6 %) difiere de esa muestra pequeña por azar y los 200 lotes bootstrap heredan la diferencia. Con `train` (951 filas, 18,6 %) o con `train` + `test` (1.155 filas, 19,8 %) son 1 de 200. Por eso la referencia es el baseline completo.
- **El bootstrap de un solo lote subestima el ruido de la referencia.** Mide la variación dentro de `monitor_green`, no entre muestras distintas del mismo origen. La cifra de 0,5 % es una cota optimista; en producción conviene revisar los ROJOS aislados antes de actuar.

---|---|---|
| 3 | _pendiente_ | _pendiente_ |

---

## 4. Retraso de etiquetas

En producción la etiqueta real (la cata) llega **días o semanas después** de la predicción. Mientras tanto no se puede calcular el F1, así que el concept drift no es visible directamente. La estrategia tiene tres capas:

1. **Señales sin etiqueta (inmediatas).** Se monitorea cada lote con:
   - el data drift de las entradas (sección 1);
   - el drift de la clase predicha (chi²);
   - la distribución de la probabilidad predicha. Un desplazamiento fuerte hacia valores intermedios indica que el modelo está menos seguro.
2. **Muestra etiquetada del 5–10 % de cada lote.** Se pide a los catadores que etiqueten de inmediato una muestra pequeña, para obtener una estimación temprana del F1 con un retraso corto. Con muestras pequeñas el F1 es muy ruidoso, por lo que esta estimación **adelanta la sospecha pero no dispara sola el reentrenamiento**.
3. **Actualización cuando llegan las etiquetas completas.** Al recibirlas se recalcula el F1 del lote, se reaplica el umbral y la persistencia de la sección 2, y se decide si corresponde reentrenar.

Consecuencia práctica: el data drift y el chi² son la alerta temprana; el F1 con etiquetas es la confirmación.

---

## 5. Requisitos cubiertos

| Requisito | Dónde |
|---|---|
| REQ-DD-02 (pruebas justificadas) | §1.2 |
| REQ-DD-03 (umbral explícito con origen) | §1.2, §1.3 |
| REQ-CD-03 (criterio de reentrenamiento) | §2.3–§2.5 |
| REQ-CD-04 (retraso de etiquetas) | §4 |
