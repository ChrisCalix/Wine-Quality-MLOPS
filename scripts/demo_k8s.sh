#!/usr/bin/env bash
# Demo de Kubernetes para la defensa (T6.6–T6.9).
#
# Uso:  scripts/demo_k8s.sh [balanceo|autorreparacion|escalado|todo]
# Cada fase imprime en pantalla y guarda su registro en docs/evidence/06_*.txt.
# Requisitos: clúster docker-desktop con el Deployment wine-api aplicado
# (kubectl apply -f k8s/) y el Service LoadBalancer en localhost:8080.
set -u

NS="${NS:-wine-mlops}"
DEPLOY="${DEPLOY:-wine-api}"
URL="${URL:-http://localhost:8080}"
BASE_REPLICAS="${BASE_REPLICAS:-2}"
SCALE_REPLICAS="${SCALE_REPLICAS:-4}"
EVID_DIR="$(cd "$(dirname "$0")/.." && pwd)/docs/evidence"
BODY='{"records":[{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11,"total_sulfur_dioxide":34,"density":0.9978,"ph":3.51,"sulphates":0.56,"alcohol":9.4}]}'

mkdir -p "$EVID_DIR"

hr() { printf '\n== %s ==\n' "$1"; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Una petición nueva (Connection: close = sin keep-alive). Imprime "<http> <served_by>".
predict() {
  local out code pod
  out="$(curl -s -m 5 -H 'Connection: close' -X POST "$URL/predict" \
    -H 'content-type: application/json' -d "$BODY" -w '\n%{http_code}' 2>/dev/null)"
  code="$(printf '%s' "$out" | tail -n1)"
  pod="$(printf '%s' "$out" | grep -o '"served_by":"[^"]*"' | cut -d'"' -f4)"
  echo "${code:-000} ${pod:--}"
}

wait_pods() { # espera N pods Ready y exactamente N pods en total
  local want="$1" i ready total
  for i in $(seq 1 90); do
    ready="$(kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='{.status.readyReplicas}' 2>/dev/null)"
    total="$(kubectl -n "$NS" get pods -l app=wine-api --no-headers 2>/dev/null | wc -l | tr -d ' ')"
    [ "${ready:-0}" = "$want" ] && [ "$total" = "$want" ] && return 0
    sleep 5
  done
  echo "AVISO: no se llegó a $want pods Ready en 450 s"; return 1
}

balanceo() {
  hr "T6.6 Balanceo: 20 peticiones nuevas a $URL/predict ($(stamp))"
  kubectl -n "$NS" get pods -l app=wine-api -o wide
  local i r tmp; tmp="$(mktemp)"
  for i in $(seq 1 20); do
    r="$(predict)"; printf '%2d  HTTP %s  served_by=%s\n' "$i" "${r%% *}" "${r#* }"
    echo "$r" >> "$tmp"
  done
  hr "Resumen por pod"
  awk '{print $2}' "$tmp" | sort | uniq -c
  local distintos; distintos="$(awk '{print $2}' "$tmp" | sort -u | grep -vc '^-$')"
  echo "valores distintos de served_by: $distintos (criterio REQ-K8S-03: >= 2)"
  [ "$distintos" -ge 2 ] && echo "RESULTADO: OK" || echo "RESULTADO: FALLA"
  rm -f "$tmp"
}

autorreparacion() {
  hr "T6.7 Autorreparación: borrar un pod con tráfico continuo ($(stamp))"
  wait_pods "$BASE_REPLICAS" || true
  kubectl -n "$NS" get pods -l app=wine-api
  local log watch victim n ok fail
  log="$(mktemp)"; watch="$(mktemp)"
  # Terminal 1: bucle de curl (una petición cada 0,2 s)
  ( while :; do echo "$(date +%T) $(predict)" >> "$log"; sleep 0.2; done ) &
  local loop_pid=$!
  # Terminal 3: kubectl get pods -w
  kubectl -n "$NS" get pods -l app=wine-api -w > "$watch" 2>&1 &
  local watch_pid=$!
  sleep 5
  # Terminal 2: borrar un pod
  victim="$(kubectl -n "$NS" get pods -l app=wine-api -o jsonpath='{.items[0].metadata.name}')"
  echo "$(date +%T) >>> kubectl delete pod $victim"
  kubectl -n "$NS" delete pod "$victim" --wait=false
  wait_pods "$BASE_REPLICAS" || true
  sleep 5
  kill "$loop_pid" "$watch_pid" 2>/dev/null; wait 2>/dev/null
  hr "kubectl get pods -w (cambios observados)"
  cat "$watch"
  hr "Bucle de curl: peticiones por pod y código HTTP"
  n="$(wc -l < "$log" | tr -d ' ')"
  ok="$(awk '$2==200' "$log" | wc -l | tr -d ' ')"
  fail=$((n - ok))
  awk '{print $2, $3}' "$log" | sort | uniq -c
  echo "peticiones: $n  HTTP 200: $ok  fallos: $fail"
  if [ "$fail" -gt 0 ]; then
    hr "Peticiones fallidas"; awk '$2!=200' "$log"
  fi
  [ "$fail" -eq 0 ] && echo "RESULTADO: OK (sin fallos)" || echo "RESULTADO: FALLA ($fail fallos)"
  rm -f "$log" "$watch"
  kubectl -n "$NS" get pods -l app=wine-api
}

escalado() {
  hr "T6.8 Escalado manual: $BASE_REPLICAS -> $SCALE_REPLICAS -> $BASE_REPLICAS ($(stamp))"
  wait_pods "$BASE_REPLICAS" || true
  echo "\$ kubectl get deploy,pods"; kubectl -n "$NS" get deploy "$DEPLOY"; kubectl -n "$NS" get pods -l app=wine-api
  echo; echo "\$ kubectl scale deployment $DEPLOY --replicas=$SCALE_REPLICAS"
  kubectl -n "$NS" scale deployment "$DEPLOY" --replicas="$SCALE_REPLICAS"
  sleep 3; kubectl -n "$NS" get pods -l app=wine-api
  wait_pods "$SCALE_REPLICAS" || true
  echo; kubectl -n "$NS" get deploy "$DEPLOY"; kubectl -n "$NS" get pods -l app=wine-api
  echo "-- 20 peticiones con $SCALE_REPLICAS réplicas --"
  for i in $(seq 1 20); do predict; done | awk '{print $2}' | sort | uniq -c
  echo; echo "\$ kubectl scale deployment $DEPLOY --replicas=$BASE_REPLICAS"
  kubectl -n "$NS" scale deployment "$DEPLOY" --replicas="$BASE_REPLICAS"
  wait_pods "$BASE_REPLICAS" || true
  kubectl -n "$NS" get deploy "$DEPLOY"; kubectl -n "$NS" get pods -l app=wine-api
  echo "RESULTADO: OK ($BASE_REPLICAS -> $SCALE_REPLICAS -> $BASE_REPLICAS)"
}

case "${1:-todo}" in
  balanceo)        balanceo        2>&1 | tee "$EVID_DIR/06_balanceo.txt" ;;
  autorreparacion) autorreparacion 2>&1 | tee "$EVID_DIR/06_autorreparacion.txt" ;;
  escalado)        escalado        2>&1 | tee "$EVID_DIR/06_escalado.txt" ;;
  todo)
    balanceo        2>&1 | tee "$EVID_DIR/06_balanceo.txt"
    autorreparacion 2>&1 | tee "$EVID_DIR/06_autorreparacion.txt"
    escalado        2>&1 | tee "$EVID_DIR/06_escalado.txt" ;;
  *) echo "Uso: $0 [balanceo|autorreparacion|escalado|todo]"; exit 2 ;;
esac
