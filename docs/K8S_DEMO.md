# Adaptación del demo de Kubernetes (G6)

Referencia: `examples/kubernetes/07-intrgation/` contiene los archivos de
namespace, deployment, service y HPA, pero los YAML están vacíos. El ejemplo
funcional `examples/kubernetes/03-service/service.yaml` muestra un Service
`ClusterIP` con selector y puerto; `05-rolling-update` y `06-recursos` aportan
la estrategia de actualización y los requests/limits.

## Qué se adopta del demo

- Un manifiesto por recurso en `k8s/`, namespace propio (`wine-mlops`).
- Selector común `app: wine-api` entre Deployment y Service.
- `POD_NAME` vía `fieldRef: metadata.name` (como en `03-service`), para que
  `served_by` muestre qué réplica respondió.
- `RollingUpdate` con `maxUnavailable: 0` y `requests/limits` (como en `05` y `06`).

## Por qué el Service es `LoadBalancer` y no `NodePort`

El Kubernetes de Docker Desktop de este equipo se aprovisiona con kind: los
nodos (`desktop-control-plane`, `desktop-worker*`) son contenedores en la red
Docker `kind` (172.18.0.0/16), que vive dentro de la VM de Docker y no se
enruta desde macOS. Los contenedores de los nodos no publican puertos, así que
un NodePort (`30080`) no es alcanzable desde `localhost` ni desde la IP del
nodo. Un Service `LoadBalancer` recibe una IP externa (`172.18.0.8`) y Docker
Desktop lo publica en `localhost:<port>`. Se usa el puerto **8080** porque el
8000 del host lo ocupa `docker compose`. Sigue repartiendo entre pods con
kube-proxy; `kubectl port-forward` no sirve para la evidencia porque fija un
solo pod. La UI consumirá el Service por su DNS interno
(`wine-api.wine-mlops.svc:8080`).

## Acceso a MLflow desde los pods

Los pods usan `http://host.docker.internal:5678` (puerto publicado de MLflow;
internamente escucha en 5000). El nombre resuelve por el DNS del clúster, sin
`hostAliases` ni IP fija. Lo que sí exige MLflow es que ese nombre esté en
`--allowed-hosts` (ver `docker-compose.yml`); sin eso rechaza el header `Host`.

## Tamaño del ejercicio

Por memoria disponible en Docker Desktop (6 GB), el ejercicio se reduce: el
Deployment base tiene **2 réplicas**, el escalado manual es **2 → 4 → 2** y el
HPA (Fase 7) va de **2 a 6**. Cada réplica carga el modelo en memoria
(límite 512Mi), y el plano de control de kind se volvió inestable con 4 GB.
Sigue habiendo ≥ 2 pods para demostrar balanceo y autorreparación.

## Robustez

- `initContainer` espera a que MLflow responda antes de arrancar la API.
- `startupProbe` (`/health`) da hasta 150 s para cargar el modelo; `readiness`
  (`/ready`) controla el tráfico y `liveness` (`/health`) reinicia si se cuelga.
- `preStop: sleep 5` y `terminationGracePeriodSeconds: 30` dan tiempo a retirar
  el pod de los endpoints antes de cerrarlo (autorreparación sin errores).

## Autoescalado (Fase 7, gate G7)

El curso no entregó ejemplo de autoescalado (`examples/kubernetes/07-intrgation/03-hpa.yaml`
está vacío), así que G7 se resolvió con las prácticas estándar de Kubernetes:

- `autoscaling/v2` con `scaleTargetRef` al Deployment y métrica `Resource` de CPU
  en `Utilization` 60 %. El porcentaje se mide sobre `requests.cpu` (100m), por
  eso los requests son obligatorios para que el cálculo sea válido.
- `minReplicas: 2` (REQ-K8S-02) y `maxReplicas: 6`, por la memoria disponible.
- `behavior.scaleUp`: máximo +2 pods cada 30 s, porque cada pod carga el modelo
  y un salto brusco agota la memoria. `behavior.scaleDown`: ventana de
  estabilización de 60 s y -1 pod cada 30 s para no oscilar.
- metrics-server v0.9.0 en `k8s/metrics-server/` con `--kubelet-insecure-tls`
  (kubelets de kind con certificado autofirmado). Se aplica antes que el HPA.
- La carga es un Job (`k8s/load-test/`) que llama al Service por su DNS interno,
  así el tráfico pasa por kube-proxy y se reparte entre pods.
- Los manifiestos de `metrics-server/` y `load-test/` están en subcarpetas para
  que `kubectl apply -f k8s/` no lance la carga ni reinstale metrics-server.

Con el HPA activo, `kubectl scale` manual deja de ser estable: el HPA reajusta
las réplicas hacia lo que pide la métrica. Las pruebas de la Fase 6
(`scripts/demo_k8s.sh`) se ejecutan sin HPA; se borra con
`kubectl -n wine-mlops delete hpa wine-api`.

