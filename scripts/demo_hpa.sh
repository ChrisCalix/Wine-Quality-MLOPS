#!/usr/bin/env bash
# Demo del HPA (T7.5): sube réplicas bajo carga y baja al terminar.
#
# Uso:  scripts/demo_hpa.sh
# Requisitos: metrics-server instalado (kubectl apply -f k8s/metrics-server/),
# Deployment wine-api aplicado y la imagen wine-api:local construida.
# Registro: docs/evidence/07_hpa.txt
set -u

NS="${NS:-wine-mlops}"
MIN="${MIN:-2}"
MAX_WAIT_UP="${MAX_WAIT_UP:-420}"     # s máximos esperando la carga
MAX_WAIT_DOWN="${MAX_WAIT_DOWN:-600}" # s máximos esperando la bajada
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/evidence/07_hpa.txt"
mkdir -p "$ROOT/docs/evidence"

hpa_line() {
  local cpu cur des ready
  cpu="$(kubectl -n "$NS" get hpa wine-api -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null)"
  cur="$(kubectl -n "$NS" get hpa wine-api -o jsonpath='{.status.currentReplicas}' 2>/dev/null)"
  des="$(kubectl -n "$NS" get hpa wine-api -o jsonpath='{.status.desiredReplicas}' 2>/dev/null)"
  ready="$(kubectl -n "$NS" get deploy wine-api -o jsonpath='{.status.readyReplicas}' 2>/dev/null)"
  printf '%s  cpu=%s%%/60%%  replicas=%s  deseadas=%s  ready=%s\n' \
    "$(date +%T)" "${cpu:-?}" "${cur:-?}" "${des:-?}" "${ready:-0}"
}

run() {
  echo "== T7.5 HPA: carga y bajada ($(date '+%Y-%m-%d %H:%M:%S')) =="
  echo "\$ kubectl apply -f k8s/hpa.yaml"
  kubectl apply -f "$ROOT/k8s/hpa.yaml"
  kubectl -n "$NS" delete job wine-load-test --ignore-not-found >/dev/null
  # estado base
  for _ in $(seq 1 60); do
    [ "$(kubectl -n "$NS" get deploy wine-api -o jsonpath='{.status.readyReplicas}')" = "$MIN" ] && break
    sleep 5
  done
  echo; echo "-- estado base --"
  kubectl -n "$NS" get hpa wine-api
  kubectl -n "$NS" top pods -l app=wine-api 2>/dev/null

  kubectl -n "$NS" get hpa wine-api -w > "$ROOT/.hpa_watch.tmp" 2>&1 &
  local watch_pid=$!

  echo; echo "\$ kubectl apply -f k8s/load-test/load-test-job.yaml"
  kubectl apply -f "$ROOT/k8s/load-test/load-test-job.yaml"
  echo; echo "-- bajo carga (hora, CPU media, réplicas) --"
  local peak=0 t=0 cur
  while [ "$t" -lt "$MAX_WAIT_UP" ]; do
    hpa_line
    cur="$(kubectl -n "$NS" get hpa wine-api -o jsonpath='{.status.currentReplicas}' 2>/dev/null)"
    [ "${cur:-0}" -gt "$peak" ] && peak="$cur"
    # termina cuando el Job acabó
    [ "$(kubectl -n "$NS" get job wine-load-test -o jsonpath='{.status.succeeded}' 2>/dev/null)" = "2" ] && break
    sleep 5; t=$((t + 5))
  done
  echo; echo "-- el Job terminó; resultado de la carga --"
  kubectl -n "$NS" logs -l app=wine-load-test --tail=2 2>/dev/null

  echo; echo "-- bajada (ventana de estabilización 60 s) --"
  t=0
  while [ "$t" -lt "$MAX_WAIT_DOWN" ]; do
    hpa_line
    cur="$(kubectl -n "$NS" get hpa wine-api -o jsonpath='{.status.currentReplicas}' 2>/dev/null)"
    [ "${cur:-99}" = "$MIN" ] && [ "$t" -gt 30 ] && break
    sleep 5; t=$((t + 5))
  done
  kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null

  echo; echo "== kubectl get hpa -w (salida cruda) =="
  cat "$ROOT/.hpa_watch.tmp"; rm -f "$ROOT/.hpa_watch.tmp"
  echo; echo "== eventos del HPA =="
  kubectl -n "$NS" describe hpa wine-api | sed -n '/^Events:/,$p'
  echo; kubectl -n "$NS" get hpa wine-api; kubectl -n "$NS" get pods -l app=wine-api
  echo
  echo "pico de réplicas: $peak (mín. $MIN, máx. 6)"
  if [ "$peak" -gt "$MIN" ] && [ "${cur:-99}" = "$MIN" ]; then
    echo "RESULTADO: OK (subió de $MIN a $peak y volvió a $MIN)"
  else
    echo "RESULTADO: FALLA (pico=$peak, final=$cur)"
  fi
}

run 2>&1 | tee "$OUT"
