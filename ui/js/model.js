// Modelo (MVC): estado del formulario y del resultado, validación y avisos de rango.
// No toca el DOM ni la red.
import { PRESET_VALUES, TRAINING_STATS } from './trainingStats.js';

// Orden y nombres de las 11 variables: contrato de la API (api/schemas.py).
export const FIELDS = [
  { name: 'fixed_acidity', label: 'Acidez fija', unit: 'g/dm³', step: 0.1 },
  { name: 'volatile_acidity', label: 'Acidez volátil', unit: 'g/dm³', step: 0.01 },
  { name: 'citric_acid', label: 'Ácido cítrico', unit: 'g/dm³', step: 0.01 },
  { name: 'residual_sugar', label: 'Azúcar residual', unit: 'g/dm³', step: 0.1 },
  { name: 'chlorides', label: 'Cloruros', unit: 'g/dm³', step: 0.001 },
  { name: 'free_sulfur_dioxide', label: 'SO₂ libre', unit: 'mg/dm³', step: 1 },
  { name: 'total_sulfur_dioxide', label: 'SO₂ total', unit: 'mg/dm³', step: 1 },
  { name: 'density', label: 'Densidad', unit: 'g/cm³', step: 0.0001 },
  { name: 'ph', label: 'pH', unit: '', step: 0.01, max: 14 },
  { name: 'sulphates', label: 'Sulfatos', unit: 'g/dm³', step: 0.01 },
  { name: 'alcohol', label: 'Alcohol', unit: '% vol', step: 0.1 },
];

export const PRESETS = {
  red: { label: 'Tinto típico', values: PRESET_VALUES.red },
  white: { label: 'Blanco típico', values: PRESET_VALUES.white },
};

export const HISTORY_SIZE = 5;

const FIELD_BY_NAME = Object.fromEntries(FIELDS.map((field) => [field.name, field]));

/** Convierte lo escrito en un número; acepta coma decimal. Devuelve null si no es válido. */
export function parseNumber(raw) {
  if (raw === null || raw === undefined) return null;
  const text = String(raw).trim().replace(',', '.');
  if (text === '' || !/^-?\d*\.?\d+(e-?\d+)?$/i.test(text)) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

/** Errores por campo según el contrato de la API: obligatorio, número, >= 0 y pH <= 14. */
export function validateValues(values) {
  const errors = {};
  for (const field of FIELDS) {
    const number = parseNumber(values[field.name]);
    if (number === null) errors[field.name] = 'Escribe un número';
    else if (number < 0) errors[field.name] = 'No puede ser negativo';
    else if (field.max !== undefined && number > field.max) errors[field.name] = `Máximo ${field.max}`;
  }
  return errors;
}

/** 'fuera' si sale de [min, max] de entrenamiento, 'atipico' si sale de [p5, p95], si no null. */
export function rangeLevel(name, value) {
  const stats = TRAINING_STATS[name];
  if (!stats || value === null) return null;
  if (value < stats.min || value > stats.max) return 'fuera';
  if (value < stats.p5 || value > stats.p95) return 'atipico';
  return null;
}

/** Avisos de rango por campo: { nombre: 'atipico' | 'fuera' }. */
export function rangeWarnings(values) {
  const warnings = {};
  for (const field of FIELDS) {
    const level = rangeLevel(field.name, parseNumber(values[field.name]));
    if (level) warnings[field.name] = level;
  }
  return warnings;
}

/** Registro numérico listo para enviar a la API. */
export function toRecord(values) {
  return Object.fromEntries(FIELDS.map((field) => [field.name, parseNumber(values[field.name])]));
}

function emptyValues() {
  return Object.fromEntries(FIELDS.map((field) => [field.name, '']));
}

export class WineModel {
  constructor() {
    this.listeners = [];
    this.state = {
      values: emptyValues(),
      touched: false, // el usuario ya intentó evaluar: se muestran los errores
      status: 'idle', // idle | loading | success | error
      result: null,
      error: null,
      modelInfo: null,
      apiStatus: 'unknown', // unknown | up | down
      preset: null,
      errors: validateValues(emptyValues()),
      warnings: {},
      history: [], // últimas evaluaciones, la más reciente primero
    };
  }

  subscribe(listener) {
    this.listeners.push(listener);
  }

  update(changes) {
    const next = { ...this.state, ...changes };
    // Estado derivado: la vista solo lo dibuja.
    next.errors = validateValues(next.values);
    next.warnings = rangeWarnings(next.values);
    this.state = next;
    this.listeners.forEach((listener) => listener(this.state));
  }

  setValue(name, raw) {
    if (!FIELD_BY_NAME[name]) return;
    this.update({ values: { ...this.state.values, [name]: raw }, preset: null });
  }

  applyPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;
    const values = Object.fromEntries(
      FIELDS.map((field) => [field.name, String(preset.values[field.name])]),
    );
    this.update({ values, preset: key, touched: false, status: 'idle', result: null, error: null });
  }

  clear() {
    this.update({
      values: emptyValues(), preset: null, touched: false, status: 'idle', result: null, error: null,
    });
  }

  /** Marca que se intentó evaluar (muestra los errores) y devuelve los errores actuales. */
  validate() {
    this.update({ touched: true });
    return this.state.errors;
  }

  setLoading() {
    this.update({ status: 'loading', error: null });
  }

  setResult(response) {
    const prediction = response.predictions[0];
    const entry = {
      at: new Date(),
      premium: prediction.premium,
      probability: prediction.probability_premium,
      pod: response.served_by,
    };
    const history = [entry, ...this.state.history].slice(0, HISTORY_SIZE);
    this.update({ status: 'success', result: response, error: null, apiStatus: 'up', history });
  }

  setError(error) {
    const down = error.kind === 'network' || error.status === 502 || error.status === 504;
    this.update({ status: 'error', error, result: null, apiStatus: down ? 'down' : this.state.apiStatus });
  }

  setModelInfo(info) {
    this.update({ modelInfo: info, apiStatus: 'up' });
  }

  setApiDown() {
    this.update({ apiStatus: 'down' });
  }
}
