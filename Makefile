PYTHON ?= python3

.PHONY: data

# Descarga ambos CSV de UCI id 186 a data/raw/ y verifica su SHA-256.
data:
	PYTHONPATH=src $(PYTHON) -m wine.data.download
