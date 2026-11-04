// Pruebas del adaptador de la API con un fetch simulado. Ejecutar: node --test ui/tests/*.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ApiClient, ApiError } from '../js/apiClient.js';

const reply = (status, body) => async () => ({ ok: status < 400, status, json: async () => body });

test('predict envía una sola muestra a /api/predict y devuelve la respuesta', async () => {
  const calls = [];
  const client = new ApiClient('/api', async (url, options) => {
    calls.push({ url, options });
    return { ok: true, status: 200, json: async () => ({ predictions: [{ premium: 1, probability_premium: 0.4 }] }) };
  });
  const response = await client.predict({ alcohol: 12 });
  assert.equal(calls[0].url, '/api/predict');
  assert.equal(calls[0].options.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].options.body), { records: [{ alcohol: 12 }] });
  assert.equal(response.predictions[0].premium, 1);
});

test('modelInfo consulta /api/model-info', async () => {
  let requested;
  const client = new ApiClient('/api', async (url) => {
    requested = url;
    return { ok: true, status: 200, json: async () => ({ model_version: '3' }) };
  });
  assert.equal((await client.modelInfo()).model_version, '3');
  assert.equal(requested, '/api/model-info');
});

test('un fallo de red se convierte en ApiError de tipo network', async () => {
  const client = new ApiClient('/api', async () => { throw new TypeError('failed to fetch'); });
  await assert.rejects(client.predict({}), (error) => error instanceof ApiError && error.kind === 'network');
});

test('un 503 conserva el estado y el mensaje de la API', async () => {
  const client = new ApiClient('/api', reply(503, { detail: 'model not loaded' }));
  await assert.rejects(client.predict({}), (error) => (
    error.kind === 'http' && error.status === 503 && error.detail === 'model not loaded'
  ));
});

test('un 422 resume los errores de validación de FastAPI', async () => {
  const body = { detail: [{ loc: ['body', 'records', 0, 'ph'], msg: 'Input should be less than or equal to 14' }] };
  const client = new ApiClient('/api', reply(422, body));
  await assert.rejects(client.predict({}), (error) => error.status === 422 && /ph/.test(error.detail));
});

test('un error sin cuerpo JSON devuelve un mensaje genérico con el estado', async () => {
  const client = new ApiClient('/api', async () => ({ ok: false, status: 502, json: async () => { throw new SyntaxError('no json'); } }));
  await assert.rejects(client.modelInfo(), (error) => error.status === 502 && /502/.test(error.detail));
});
