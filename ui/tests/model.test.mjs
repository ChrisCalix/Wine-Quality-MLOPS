// Pruebas del modelo MVC (funciones puras y estado). Ejecutar: node --test ui/tests/*.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  FIELDS, HISTORY_SIZE, PRESETS, WineModel, parseNumber, rangeLevel, rangeWarnings, toRecord, validateValues,
} from '../js/model.js';

const FEATURES_API = [
  'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
  'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol',
];

const asStrings = (values) => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, String(v)]));

test('los campos siguen el orden y los nombres del contrato de la API', () => {
  assert.deepEqual(FIELDS.map((field) => field.name), FEATURES_API);
});

test('parseNumber acepta coma decimal y rechaza texto', () => {
  assert.equal(parseNumber('7,4'), 7.4);
  assert.equal(parseNumber(' 0.9978 '), 0.9978);
  assert.equal(parseNumber('abc'), null);
  assert.equal(parseNumber(''), null);
  assert.equal(parseNumber('1.2.3'), null);
});

test('validateValues exige número, no negativo y pH <= 14', () => {
  const ok = asStrings(PRESETS.red.values);
  assert.deepEqual(validateValues(ok), {});
  assert.match(validateValues({ ...ok, alcohol: '' }).alcohol, /número/);
  assert.match(validateValues({ ...ok, chlorides: '-0.1' }).chlorides, /negativo/);
  assert.match(validateValues({ ...ok, ph: '20' }).ph, /Máximo 14/);
});

test('rangeLevel distingue atípico (p5-p95) y fuera (min-max)', () => {
  assert.equal(rangeLevel('alcohol', 10.1), null);
  assert.equal(rangeLevel('alcohol', 13), 'atipico');
  assert.equal(rangeLevel('alcohol', 20), 'fuera');
  assert.equal(rangeLevel('alcohol', null), null);
});

test('el preset de tinto no tiene avisos y el de blanco marca variables atípicas', () => {
  assert.deepEqual(rangeWarnings(asStrings(PRESETS.red.values)), {});
  const white = rangeWarnings(asStrings(PRESETS.white.values));
  assert.ok(Object.keys(white).length >= 1);
  assert.ok('total_sulfur_dioxide' in white);
});

test('toRecord devuelve números listos para la API', () => {
  const record = toRecord(asStrings(PRESETS.red.values));
  assert.deepEqual(Object.keys(record), FEATURES_API);
  assert.ok(Object.values(record).every((value) => typeof value === 'number'));
});

test('el modelo aplica presets, valida y notifica a los suscriptores', () => {
  const model = new WineModel();
  let notified = 0;
  model.subscribe(() => { notified += 1; });
  assert.equal(Object.keys(model.state.errors).length, 11);

  model.applyPreset('white');
  assert.equal(model.state.preset, 'white');
  assert.equal(Object.keys(model.state.errors).length, 0);
  assert.ok(Object.keys(model.state.warnings).length >= 1);

  model.setValue('alcohol', '');
  assert.equal(model.state.preset, null);
  assert.ok(model.validate().alcohol);
  assert.equal(model.state.touched, true);
  assert.ok(notified >= 3);

  model.clear();
  assert.equal(model.state.touched, false);
  assert.equal(model.state.values.alcohol, '');
});

test('los errores de red marcan la API como caída y un resultado la marca como activa', () => {
  const model = new WineModel();
  model.setError({ kind: 'network', status: 0, detail: 'sin red' });
  assert.equal(model.state.apiStatus, 'down');
  assert.equal(model.state.status, 'error');
  model.setResult({ predictions: [{ premium: 0, probability_premium: 0.1 }] });
  assert.equal(model.state.apiStatus, 'up');
  assert.equal(model.state.status, 'success');
});

test('el historial guarda las últimas evaluaciones, la más reciente primero', () => {
  const model = new WineModel();
  const respond = (pod, probability) => ({
    predictions: [{ premium: probability >= 0.22 ? 1 : 0, probability_premium: probability }],
    served_by: pod,
  });
  for (let i = 0; i < HISTORY_SIZE + 3; i += 1) model.setResult(respond(`pod-${i}`, i / 10));
  const { history } = model.state;
  assert.equal(history.length, HISTORY_SIZE);
  assert.equal(history[0].pod, `pod-${HISTORY_SIZE + 2}`);
  assert.equal(history[0].premium, 1);
  model.clear();
  assert.equal(model.state.history.length, HISTORY_SIZE);
});
