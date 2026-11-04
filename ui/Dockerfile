# UI estática servida por nginx (imagen oficial sin privilegios, versión fijada).
# Contexto de construcción: la carpeta ui/   ->   docker build -t wine-ui:local ui/
FROM nginxinc/nginx-unprivileged:1.28.2-alpine

# Valores por defecto del clúster; se pueden sobrescribir al ejecutar (docker run -e ...).
ENV DNS_RESOLVER=kube-dns.kube-system.svc.cluster.local \
    API_UPSTREAM=http://wine-api.wine-mlops.svc.cluster.local:8080

# La imagen genera /etc/nginx/conf.d/default.conf a partir de esta plantilla al arrancar.
COPY nginx.conf /etc/nginx/templates/default.conf.template

COPY index.html /usr/share/nginx/html/index.html
COPY css/ /usr/share/nginx/html/css/
COPY js/ /usr/share/nginx/html/js/

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
