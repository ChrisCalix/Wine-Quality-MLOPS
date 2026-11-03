# Data card — Wine Quality (UCI id 186)

## Origen

| Campo | Valor |
|---|---|
| Nombre | Wine Quality |
| Autores | P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009) |
| Repositorio | UCI Machine Learning Repository, dataset id 186: https://archive.ics.uci.edu/dataset/186/wine+quality |
| DOI | 10.24432/C56S3T |
| Licencia | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Artículo | "Modeling wine preferences by data mining from physicochemical properties", *Decision Support Systems*, 2009 |
| Descarga | `make data` (`src/wine/data/download.py`), desde `archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/` |

Cita sugerida: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

## Archivos y versión

La versión queda fijada por el SHA-256 de cada archivo, que `download.py` verifica en cada descarga (si el origen cambia, el comando falla). Dos descargas consecutivas dieron los mismos hashes.

| Archivo | Filas | SHA-256 |
|---|---|---|
| `winequality-red.csv` | 1.599 | `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e` |
| `winequality-white.csv` | 4.898 | `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836` |

Total: 6.497 registros. Separador `;`, 11 variables de entrada y la columna `quality`. Los CSV viven en `data/raw/`, que git ignora.

## Variables de entrada

| Nombre en el CSV | Nombre interno | Unidad |
|---|---|---|
| `fixed acidity` | `fixed_acidity` | g/dm³ (ácido tartárico) |
| `volatile acidity` | `volatile_acidity` | g/dm³ (ácido acético) |
| `citric acid` | `citric_acid` | g/dm³ |
| `residual sugar` | `residual_sugar` | g/dm³ |
| `chlorides` | `chlorides` | g/dm³ (cloruro de sodio) |
| `free sulfur dioxide` | `free_sulfur_dioxide` | mg/dm³ |
| `total sulfur dioxide` | `total_sulfur_dioxide` | mg/dm³ |
| `density` | `density` | g/cm³ |
| `pH` | `ph` | — |
| `sulphates` | `sulphates` | g/dm³ (sulfato de potasio) |
| `alcohol` | `alcohol` | % vol |

El objetivo original `quality` es un entero (mediana de al menos tres catadores). El objetivo del proyecto es `premium = 1` si `quality >= 7`.

## Observaciones sobre los archivos descargados

Medidas sobre los CSV con hash fijado, antes de limpiar:

| Medida | Tintos | Blancos |
|---|---|---|
| Celdas vacías | 0 | 0 |
| Filas duplicadas exactas (12 columnas) | 240 | 937 |
| Filas con `quality >= 7` | 217 (13,6 %) | 1.060 (21,6 %) |

La deduplicación y la tasa de premium tras limpiar se reportarán con el ETL (Fase 3).

## Advertencias

- **No usar `sklearn.datasets.load_wine()`:** son 178 registros de otro problema (cultivar de vino italiano).
- **No usar variantes de Kaggle como WineQT:** es un subconjunto no oficial, con columna `Id`, sin trazabilidad.
- Todos los vinos son *vinho verde* del norte de Portugal; el modelo no debe extrapolarse a otras denominaciones.
- La etiqueta `quality` refleja el criterio subjetivo de los catadores.
- `wine_type` (tinto/blanco) se agrega solo para análisis y nunca es variable de entrada del modelo.
