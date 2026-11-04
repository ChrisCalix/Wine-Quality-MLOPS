#!/usr/bin/env bash
# CD local: actualiza el Deployment wine-api del clúster de Docker Desktop con la imagen
# que publicó el workflow CD en ghcr.io, y espera a que el rollout termine.
#
# Uso:  scripts/deploy_local.sh [SHA]
#   SHA       commit cuya imagen se despliega (por defecto, el HEAD actual)
#   IMAGE     imagen completa; sustituye a ghcr.io/<repo>:<SHA> (p. ej. una imagen local)
#   NS, DEPLOY, URL, TIMEOUT   namespace, deployment, URL del Service y espera en segundos
#
# Si el rollout falla (la imagen no existe, no arranca o nunca queda lista), revierte al
# ReplicaSet anterior: con maxUnavailable: 0 los pods viejos siguen sirviendo mientras tanto.
# La imagen de ghcr.io debe ser pública (GitHub > Packages > Package settings > Visibility).
set -euo pipefail

NS="${NS:-wine-mlops}"
DEPLOY="${DEPLOY:-wine-api}"
URL="${URL:-http://localhost:8080}"
TIMEOUT="${TIMEOUT:-300}"

REF="${1:-$(git rev-parse HEAD)}"
REPO="$(git remote get-url origin | sed -E 's#^(git@github.com:|https://github.com/)##; s#\.git$##' | tr '[:upper:]' '[:lower:]')"
IMAGE="${IMAGE:-ghcr.io/${REPO}:${REF}}"

echo "== CD local: $DEPLOY (namespace $NS)"
echo "   imagen: $IMAGE"
echo "   antes:  $(kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].image}')"

# El initContainer usa la misma imagen que el contenedor de la API.
kubectl -n "$NS" set image "deploy/$DEPLOY" "api=$IMAGE" "wait-for-mlflow=$IMAGE"

if ! kubectl -n "$NS" rollout status "deploy/$DEPLOY" --timeout="${TIMEOUT}s"; then
  echo "== El rollout falló: se revierte a la versión anterior" >&2
  kubectl -n "$NS" rollout undo "deploy/$DEPLOY"
  kubectl -n "$NS" rollout status "deploy/$DEPLOY" --timeout=120s || true
  exit 1
fi

echo "== Desplegado"
echo "   ahora:  $(kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].image}')"
kubectl -n "$NS" get pods -l app=wine-api -o wide
echo "== $URL/model-info"
curl -s --max-time 10 "$URL/model-info" || echo "(sin respuesta)"
echo
