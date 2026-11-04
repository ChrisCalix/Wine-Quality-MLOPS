// Adaptador de la API: única pieza que conoce la URL base ("/api") y el formato de las respuestas.
// En el clúster, nginx reenvía "/api/" al Service wine-api; en local lo hace ui/dev_server.py.

export class ApiError extends Error {
  /** kind: 'network' (no hubo respuesta) | 'http' (la API respondió con error). */
  constructor(kind, status, detail) {
    super(detail);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

/** Extrae un mensaje legible del cuerpo de error de FastAPI ({"detail": ...}). */
function describe(body, status) {
  const detail = body && body.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => `${(item.loc || []).slice(1).join('.')}: ${item.msg}`).join('; ');
  }
  return `La API respondió con el estado ${status}`;
}

export class ApiClient {
  constructor(baseUrl = '/api', fetchImpl = (...args) => fetch(...args)) {
    this.baseUrl = baseUrl;
    this.fetchImpl = fetchImpl;
  }

  async request(path, options = {}) {
    let response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      });
    } catch {
      throw new ApiError('network', 0, 'No se pudo contactar con la API');
    }
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (!response.ok) throw new ApiError('http', response.status, describe(body, response.status));
    return body;
  }

  /** Evalúa una muestra; devuelve la respuesta completa de /predict. */
  predict(record) {
    return this.request('/predict', { method: 'POST', body: JSON.stringify({ records: [record] }) });
  }

  modelInfo() {
    return this.request('/model-info');
  }
}
