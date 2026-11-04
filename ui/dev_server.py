"""Servidor de desarrollo de la UI: sirve ``ui/`` y reenvía ``/api/*`` a la API.

En el clúster ese reenvío lo hace nginx (``ui/nginx.conf``); aquí se hace igual para que el
navegador llame siempre a ``/api`` en el mismo origen, sin CORS.

    python ui/dev_server.py                       # UI en http://localhost:8082
    API_URL=http://localhost:8000 python ui/dev_server.py
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")
PORT = int(os.environ.get("PORT", "8082"))
UI_DIR = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        request = urllib.request.Request(
            API_URL + self.path[len("/api"):],
            data=body,
            method=self.command,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, payload = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, payload = error.code, error.read()
        except OSError as error:
            status = 502
            payload = f'{{"detail": "No se pudo contactar con la API en {API_URL}: {error}"}}'.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._proxy() if self.path.startswith("/api/") else super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/"):
            self._proxy()
        else:
            self.send_error(405)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # desarrollo: ver siempre la última versión
        super().end_headers()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(UI_DIR)))
    print(f"UI en http://localhost:{PORT}  (API: {API_URL})")
    server.serve_forever()
