// GENERADO por scripts/gen_ui_stats.py; no editar a mano.
// Rangos: tintos de data/processed/train.csv (951 filas).
// Presets: mediana de los tintos de train y de la muestra de blancos (500 filas).

export const TRAINING_STATS = {
  fixed_acidity: { min: 4.7, p5: 6.1, p95: 11.9, max: 15.9 },
  volatile_acidity: { min: 0.12, p5: 0.27, p95: 0.8375, max: 1.33 },
  citric_acid: { min: 0, p5: 0, p95: 0.615, max: 0.78 },
  residual_sugar: { min: 1.2, p5: 1.6, p95: 4.55, max: 15.5 },
  chlorides: { min: 0.012, p5: 0.052, p95: 0.1455, max: 0.611 },
  free_sulfur_dioxide: { min: 1, p5: 4, p95: 35.5, max: 72 },
  total_sulfur_dioxide: { min: 6, p5: 11, p95: 119, max: 289 },
  density: { min: 0.9902, p5: 0.9936, p95: 0.9998, max: 1.0037 },
  ph: { min: 2.86, p5: 3.05, p95: 3.56, max: 4.01 },
  sulphates: { min: 0.33, p5: 0.47, p95: 0.965, max: 1.98 },
  alcohol: { min: 8.4, p5: 9.2, p95: 12.5, max: 14.9 },
};

export const PRESET_VALUES = {
  red: { fixed_acidity: 7.9, volatile_acidity: 0.52, citric_acid: 0.26, residual_sugar: 2.2, chlorides: 0.079, free_sulfur_dioxide: 14, total_sulfur_dioxide: 39, density: 0.9968, ph: 3.3, sulphates: 0.62, alcohol: 10.1 },
  white: { fixed_acidity: 6.8, volatile_acidity: 0.27, citric_acid: 0.32, residual_sugar: 4.775, chlorides: 0.044, free_sulfur_dioxide: 33, total_sulfur_dioxide: 135, density: 0.9937, ph: 3.185, sulphates: 0.46, alcohol: 10.4 },
};
