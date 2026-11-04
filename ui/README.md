# UI — Wine Quality

Página web (HTML, Bootstrap 5 y JavaScript con módulos ES, sin dependencias de compilación) para evaluar si una muestra de vino tinto tiene potencial premium. Llama al API de inferencia y muestra la probabilidad, la versión del modelo y el pod que respondió.

## Cómo verla

```bash
python ui/dev_server.py        # http://localhost:8082
```

El servidor de desarrollo sirve `ui/` y reenvía `/api/*` a la API (por defecto `http://localhost:8080`, el Service del clúster). Se cambian con `API_URL` y `PORT`. En el clúster ese reenvío lo hace nginx (`ui/nginx.conf`), de modo que el navegador siempre llama a `/api` en su mismo origen y no hay CORS.

## En el clúster (contenedor)

La UI también corre en Kubernetes como un contenedor de nginx, sin Node ni paso de compilación:

| Pieza | Archivo | Qué hace |
|---|---|---|
| Imagen | `ui/Dockerfile` | `nginxinc/nginx-unprivileged:1.28.2-alpine` (usuario no root, versión fijada) con solo `index.html`, `css/` y `js/` |
| nginx | `ui/nginx.conf` (plantilla) | Sirve la UI, responde `/healthz` y reenvía `/api/` a `http://wine-api.wine-mlops.svc.cluster.local:8080` |
| Despliegue | `k8s/ui/deployment.yaml` | 2 réplicas, sondas sobre `/healthz`, límites pequeños y `runAsNonRoot` |
| Acceso | `k8s/ui/service.yaml` | Service `wine-ui` de tipo LoadBalancer: **http://localhost:8081** |

El camino de una evaluación es: navegador → nginx de la UI → Service `wine-api` → pods de la API → modelo `champion` de MLflow. nginx resuelve el Service por DNS interno (CoreDNS) y el código de la UI nunca menciona `localhost`.

```bash
docker build -t wine-ui:local ui/            # construir la imagen
kubectl apply -f k8s/ui/                     # desplegar (namespace wine-mlops)
COMPONENT=ui IMAGE=wine-ui:local scripts/deploy_local.sh   # o actualizar el Deployment existente
```

`API_UPSTREAM` y `DNS_RESOLVER` son variables de la imagen (por defecto, las del clúster), de modo que el mismo contenedor se puede probar fuera de Kubernetes apuntando a otra API:

```bash
docker run --rm -p 8083:8080 -e DNS_RESOLVER=<dns del contenedor> -e API_UPSTREAM=http://host.docker.internal:8080 wine-ui:local
```

## Patrón de diseño: MVC ligero con un adaptador

```
ui/
├── index.html            # esqueleto de la página y Bootstrap por CDN (con SRI)
├── css/styles.css        # color de marca y barra de probabilidad
└── js/
    ├── main.js           # crea las piezas y las conecta
    ├── model.js          # M: estado, validación y avisos de rango (sin DOM ni red)
    ├── view.js           # V: dibuja el estado con componentes de Bootstrap
    ├── controller.js     # C: eventos de la vista → modelo y API
    ├── apiClient.js      # adaptador: única pieza que conoce "/api" y el formato de las respuestas
    └── trainingStats.js  # GENERADO: rangos de entrenamiento y presets
```

Flujo: la **vista** emite eventos (escribir, elegir un preset, evaluar); el **controlador** los traduce en cambios del **modelo** o en llamadas al **adaptador**; el modelo notifica a sus suscriptores y la vista se redibuja con el nuevo estado. La vista no valida ni llama a la red, y el modelo no toca el DOM, por eso el modelo y el adaptador se prueban con Node sin navegador.

**Por qué un adaptador.** Si cambia la URL, el formato de la respuesta o se sustituye la API, solo se toca `apiClient.js` (REQ-UI-03). La UI no conoce el nombre del Service de Kubernetes ni su puerto.

## Contrato con la API

| Acción | Llamada | Uso en la UI |
|---|---|---|
| Evaluar una muestra | `POST /api/predict` con `{"records": [ {11 variables} ]}` | Clase, probabilidad, versión, alias, `run_id` y `served_by` |
| Estado del modelo | `GET /api/model-info` | Insignia de la cabecera y umbral de decisión (`decision_threshold`) |

Las once variables van en `snake_case` y en el orden de `api/schemas.py`. La clase `premium` ya viene decidida por la API con el umbral de la versión; la UI solo dibuja la probabilidad con una marca en ese umbral.

## Estados que maneja

- **Sin resultado:** mensaje inicial.
- **Validación:** cada campo es obligatorio, numérico (acepta coma decimal) y no negativo; el pH no puede pasar de 14. Los errores salen al intentar evaluar y el foco va al primero.
- **Cargando:** indicador y botón deshabilitado.
- **Error de red o de la API:** mensaje según el caso (sin conexión, modelo todavía no listo, datos rechazados, API que no responde). La insignia de la cabecera pasa a «API sin conexión» y se reintenta cada 10 s.
- **Diseño responsive:** una columna por debajo de 992 px; probado a 375 px.

## Aviso de «fuera del rango de entrenamiento»

Cada campo se compara con los tintos de `train`:

| Nivel | Condición | Señal |
|---|---|---|
| Atípico | Fuera de los percentiles 5–95 | Borde ámbar y mensaje bajo el campo |
| Fuera del rango | Fuera de mínimo–máximo | Borde rojo y mensaje bajo el campo |

Y el resultado resume cuántas variables son atípicas. **Es una pista por muestra, no un detector de drift.** Con el preset «blanco típico» (medianas de la muestra de blancos) solo 4 de las 11 variables salen atípicas y ninguna sale del mínimo–máximo de los tintos: un blanco típico cae dentro de los rangos de los tintos variable por variable. Lo que sí cambia es la distribución del lote completo, y eso lo detecta el monitoreo de data drift (`docs/DRIFT.md`), no esta pantalla.

El modelo puede marcar «premium» una muestra de blanco: solo conoce tintos. Por eso el aviso dice que su estimación es menos fiable.

## Regenerar los rangos y los presets

`js/trainingStats.js` se genera desde los datos; no se edita a mano. Si cambian los splits o se reentrena:

```bash
PYTHONPATH=src python scripts/gen_ui_stats.py
```

## Pruebas

```bash
node --test ui/tests/*.test.mjs
```

Cubren el modelo (parseo, validación, rangos, presets, estado) y el adaptador (petición, errores de red, 503, 422 y respuestas sin JSON). El CI las ejecuta en cada pull request.
