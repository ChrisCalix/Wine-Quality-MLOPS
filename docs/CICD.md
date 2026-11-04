# CI/CD — Wine Quality MLOps

Referencia del curso: `examples/heladeria-ci-cd-ct/` (GitHub Actions). Esta página anota lo que se adopta, lo que se adapta y por qué (gate G8).

## Lo que dice el ejemplo

| Tema | En el ejemplo |
|---|---|
| Plataforma | GitHub Actions (`.github/workflows/`) |
| Registro de imágenes | `ghcr.io/<usuario>/<repositorio>` |
| Autenticación | `GITHUB_TOKEN` del propio workflow con `permissions: packages: write`; **sin secretos** |
| CI | En cada push a `main` y en cada pull request: verifica datos y corre `pytest` |
| CD | Después del CI, solo en `main`: construye la imagen, la **prueba** (levanta el contenedor y pide una predicción) y la publica con dos etiquetas, el SHA del commit y `latest` |
| CT | Programado o manual: revisa drift y, si hay concept drift, reentrena, guarda y lanza el CI/CD |
| Despliegue | Llega hasta publicar la imagen; **no actualiza ningún clúster** |

## Qué se adopta

- GitHub Actions, con las mismas versiones de acciones que usa el ejemplo (`actions/checkout@v7`, `actions/setup-python@v7`, `docker/login-action@v4`).
- Ejecutar el CI en cada pull request y en cada push a `main`.
- Instalar las dependencias fijadas del repositorio, con caché de `pip`.
- Imagen etiquetada con el SHA del commit (REQ-CICD-02, a cargo de Dev B).
- Publicar en `ghcr.io` con el `GITHUB_TOKEN`, sin secretos en el repositorio.
- Probar la imagen antes de publicarla.

## Qué se adapta

| Tema | Decisión | Motivo |
|---|---|---|
| Estructura de workflows | `ci.yml` y `cd.yml` en archivos separados | Cada uno hace una cosa y el CD puede reaccionar al resultado del CI |
| Encadenado CI → CD | `cd.yml` se dispara con `workflow_run` del CI sobre `main` | Un workflow lanzado con el `GITHUB_TOKEN` no dispara a los demás; un push humano a `main` sí lanza el CI y, al terminar, el CD |
| Pruebas del CI | `ruff`, `pytest` (ETL, API con modelo simulado, data drift, concept drift), búsqueda de `pickle`/`joblib` y `docker build` | Son las del plan (Fase 8) y REQ-CICD-01, REQ-MLF-01 |
| Gate de drift | La CLI `python -m wine.drift.data` debe devolver 0 con `monitor_green` y 1 con `monitor_red_white` | Es lo que permite que el CI bloquee con datos fuera de distribución |
| Servidor MLflow | El CI **no** necesita MLflow: las pruebas usan modelos locales y simulados, y el gate de drift omite el chi² cuando no hay `MLFLOW_TRACKING_URI` | Evita depender de un servicio externo en CI |
| Despliegue | El ejemplo se detiene en la imagen; aquí un script local actualiza el Deployment `wine-api` en el clúster | Es lo que pide REQ-CICD-02 y el clúster no es alcanzable desde GitHub (ver la sección siguiente) |

## CD: cómo se despliega en el clúster local (decidido)

El clúster es el Kubernetes de Docker Desktop de una Mac y un runner de GitHub no lo alcanza. Por eso el CD tiene dos jobs, y el segundo corre en un **runner autohospedado** en esa Mac:

| Job | Dónde corre | Qué hace |
|---|---|---|
| `publicar` | Runner de GitHub (`ubuntu-latest`) | Tras un CI en verde sobre `main`: construye la imagen, la prueba (`/health` 200 y `/ready` 503 sin MLflow) y la publica en `ghcr.io/<repo en minúsculas>` con el SHA del commit y `latest` |
| `desplegar` | Runner autohospedado en la Mac (`[self-hosted, macOS, ARM64]`) | Ejecuta `scripts/deploy_local.sh <SHA>`: `kubectl set image` en el Deployment `wine-api` (contenedor e initContainer), `rollout status` y comprobación de `/model-info`. Si el rollout falla, revierte solo |

El script sigue sirviendo para desplegar a mano (`scripts/deploy_local.sh [SHA]`, o con `IMAGE=...` para una imagen local).

Detalles que importan:

- **Minúsculas:** `ghcr.io` rechaza nombres con mayúsculas y este repositorio es `ChrisCalix/Wine-Quality-MLOPS`, así que el workflow y el script lo convierten con `tr`.
- **`workflow_run`:** el CD reacciona a que el CI termine en `main` (un push humano). Esto evita el problema del ejemplo: un workflow lanzado con el `GITHUB_TOKEN` no dispara a otros.
- **Paquete público:** para que los nodos descarguen la imagen sin credenciales, hay que poner el paquete en **público** una sola vez, después de la primera publicación (perfil, Packages, el paquete, *Package settings*, *Change visibility*).
- **`PATH` del servicio:** el runner arranca con un `PATH` mínimo, así que el job añade `/usr/local/bin`, donde están `kubectl` y `docker`.
- **Probado en local** contra el clúster real: despliegue correcto con rolling update sin cortes, y despliegue con una imagen inexistente, que revierte y deja el servicio respondiendo.

### Seguridad del runner

Un runner autohospedado ejecuta código del repositorio en tu Mac, y este repositorio es público. Por eso:

- El CI (que sí corre en pull requests) usa `ubuntu-latest`, nunca el runner de la Mac.
- Solo el job `desplegar` usa el runner, y solo se dispara por un CI terminado en `main` o por `workflow_dispatch`, que únicamente pueden lanzar personas con permiso de escritura.
- Ningún job de GitHub cae en la Mac por accidente: solo los que piden explícitamente `self-hosted`, y hoy es únicamente `desplegar`.
- Si se quiere más seguridad: *Settings, Actions, General, Fork pull request workflows*, exigir aprobación para colaboradores externos.

### Registrar el runner (una vez)

1. En GitHub: repositorio, *Settings*, *Actions*, *Runners*, *New self-hosted runner*; elegir **macOS** y **ARM64**. La página muestra los comandos exactos de descarga y el token de registro (caduca en una hora); no se guarda en el repositorio.
2. En la Mac, en una carpeta **fuera del repositorio**, ejecutar los comandos de descarga que muestra GitHub y registrar el runner (Enter en las preguntas: grupo, nombre, etiquetas y carpeta de trabajo por defecto):

   ```bash
   mkdir ~/actions-runner && cd ~/actions-runner
   # ...descargar y descomprimir según indica GitHub...
   ./config.sh --url https://github.com/ChrisCalix/Wine-Quality-MLOPS --token <TOKEN>
   ```

3. Dejarlo encendido: en primer plano con `./run.sh`, o como servicio de macOS que arranca solo:

   ```bash
   ./svc.sh install
   ./svc.sh start
   ./svc.sh status
   ```

4. Comprobar en *Settings, Actions, Runners* que aparece como **Idle**.

Si el runner está apagado cuando termina el CI, el job `desplegar` queda esperando; la imagen ya está publicada y se puede desplegar a mano con el script.

## Cómo se ejecuta cada paso en local

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check src tests api
pytest -v
PYTHONPATH=src python -m wine.drift.data --batch data/processed/monitor_green.csv       # código 0
PYTHONPATH=src python -m wine.drift.data --batch data/processed/monitor_red_white.csv   # código 1
docker build -t wine-api:ci .
```
