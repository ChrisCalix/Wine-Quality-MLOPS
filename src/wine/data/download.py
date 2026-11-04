"""Descarga los CSV oficiales de Wine Quality (UCI id 186) y verifica su SHA-256.

Solo usa la biblioteca estándar. Ejecutar desde la raíz del repositorio:

    PYTHONPATH=src python -m wine.data.download
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Final

BASE_URL: Final[str] = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
)
RAW_DIR: Final[Path] = Path("data/raw")

# SHA-256 de la versión oficial de cada archivo. Cualquier cambio en el origen
# hace fallar la descarga en lugar de entrenar con datos distintos en silencio.
EXPECTED_SHA256: Final[dict[str, str]] = {
    "winequality-red.csv": (
        "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
    ),
    "winequality-white.csv": (
        "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, directory: Path = RAW_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
        partial.write_bytes(response.read())
    partial.replace(target)
    return target


def main() -> int:
    failed = False
    for name, expected in EXPECTED_SHA256.items():
        path = download(name)
        digest = sha256_of(path)
        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        status = "sin hash fijado" if not expected else (
            "ok" if digest == expected else "HASH DISTINTO"
        )
        print(f"{name}  filas={rows}  sha256={digest}  [{status}]")
        failed = failed or (bool(expected) and digest != expected)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
