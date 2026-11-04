// Vista (MVC): dibuja el estado en el DOM con componentes de Bootstrap 5.
// No valida ni llama a la API; solo emite los eventos del usuario.
import { FIELDS, PRESETS } from './model.js';
import { TRAINING_STATS } from './trainingStats.js';

const nf = new Intl.NumberFormat('es', { maximumFractionDigits: 4 });
const pct = (p, digits = 1) => `${(p * 100).toLocaleString('es', { maximumFractionDigits: digits })} %`;

function el(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export class View {
  constructor(root = document) {
    this.fieldsHost = root.getElementById('fields');
    this.form = root.getElementById('wine-form');
    this.result = root.getElementById('result');
    this.historyHost = root.getElementById('history');
    this.apiBadge = root.getElementById('api-badge');
    this.presetBar = root.getElementById('preset-bar');
    this.clearButton = root.getElementById('btn-clear');
    this.submitButton = root.getElementById('btn-evaluate');
    this.inputs = {};
    this.messages = {};
  }

  buildForm() {
    for (const field of FIELDS) {
      const col = el('div', 'col');
      const label = el('label', 'form-label fw-semibold mb-1', field.label);
      label.htmlFor = `f-${field.name}`;
      const group = el('div', 'input-group');
      const input = el('input', 'form-control');
      Object.assign(input, {
        id: `f-${field.name}`, name: field.name, type: 'text', inputMode: 'decimal',
        autocomplete: 'off', placeholder: '0',
      });
      input.setAttribute('aria-describedby', `m-${field.name}`);
      group.append(input);
      if (field.unit) group.append(el('span', 'input-group-text unit', field.unit));
      const message = el('div', 'form-text field-msg');
      message.id = `m-${field.name}`;
      col.append(label, group, message);
      this.fieldsHost.append(col);
      this.inputs[field.name] = input;
      this.messages[field.name] = message;
    }
    for (const [key, preset] of Object.entries(PRESETS)) {
      const button = el('button', 'btn btn-outline-secondary btn-sm', preset.label);
      Object.assign(button, { type: 'button' });
      button.dataset.preset = key;
      this.presetBar.append(button);
    }
  }

  // --- eventos del usuario -------------------------------------------------
  onInput(handler) {
    this.fieldsHost.addEventListener('input', (event) => handler(event.target.name, event.target.value));
  }

  onPreset(handler) {
    this.presetBar.addEventListener('click', (event) => {
      const key = event.target.closest('[data-preset]')?.dataset.preset;
      if (key) handler(key);
    });
  }

  onClear(handler) {
    this.clearButton.addEventListener('click', handler);
  }

  onSubmit(handler) {
    this.form.addEventListener('submit', (event) => {
      event.preventDefault();
      handler();
    });
  }

  focusFirstError(errors) {
    const first = FIELDS.find((field) => errors[field.name]);
    if (first) this.inputs[first.name].focus();
  }

  // --- dibujo ---------------------------------------------------------------
  render(state) {
    this.renderApiBadge(state);
    this.renderFields(state);
    for (const button of this.presetBar.querySelectorAll('[data-preset]')) {
      const active = button.dataset.preset === state.preset;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    this.submitButton.disabled = state.status === 'loading';
    this.renderResult(state);
    this.renderHistory(state);
  }

  renderApiBadge({ apiStatus, modelInfo }) {
    const badge = this.apiBadge;
    badge.className = 'badge rounded-pill';
    if (apiStatus === 'up') {
      badge.classList.add('text-bg-success');
      badge.textContent = modelInfo
        ? `API conectada · modelo v${modelInfo.model_version} (${modelInfo.model_alias})`
        : 'API conectada';
    } else if (apiStatus === 'down') {
      badge.classList.add('text-bg-danger');
      badge.textContent = 'API sin conexión';
    } else {
      badge.classList.add('text-bg-secondary');
      badge.textContent = 'Conectando…';
    }
  }

  renderFields({ values, errors, warnings, touched }) {
    for (const field of FIELDS) {
      const input = this.inputs[field.name];
      if (input.value !== values[field.name]) input.value = values[field.name];
      const showError = errors[field.name] && (touched || values[field.name] !== '');
      const warning = !errors[field.name] ? warnings[field.name] : null;
      input.classList.toggle('is-invalid', Boolean(showError));
      input.classList.toggle('border-warning', warning === 'atipico');
      input.classList.toggle('border-danger', warning === 'fuera');
      input.setAttribute('aria-invalid', String(Boolean(showError)));
      const message = this.messages[field.name];
      message.className = 'form-text field-msg';
      const stats = TRAINING_STATS[field.name];
      if (showError) {
        message.textContent = errors[field.name];
        message.classList.add('text-danger');
      } else if (warning === 'fuera') {
        message.textContent = `Fuera del rango de entrenamiento (${nf.format(stats.min)}–${nf.format(stats.max)})`;
        message.classList.add('text-danger');
      } else if (warning === 'atipico') {
        message.textContent = `Atípico: el 90 % de los tintos está entre ${nf.format(stats.p5)} y ${nf.format(stats.p95)}`;
        message.classList.add('text-warning-emphasis');
      } else {
        message.textContent = '';
      }
    }
  }

  renderResult(state) {
    const host = this.result;
    host.replaceChildren();
    if (state.status === 'loading') {
      const box = el('div', 'text-center text-body-secondary py-5');
      const spinner = el('div', 'spinner-border text-secondary mb-3');
      spinner.setAttribute('role', 'status');
      box.append(spinner, el('div', '', 'Evaluando la muestra…'));
      host.append(box);
    } else if (state.status === 'error') {
      host.append(this.errorView(state.error));
    } else if (state.status === 'success' && state.result) {
      host.append(...this.successView(state));
    } else {
      const box = el('div', 'text-center text-body-secondary py-5 px-3');
      box.append(
        el('div', 'fs-5 mb-2', 'Sin resultado todavía'),
        el('div', '', 'Elige un preset o escribe el análisis de laboratorio y pulsa «Evaluar».'),
      );
      host.append(box);
    }
  }

  renderHistory({ history }) {
    const host = this.historyHost;
    host.replaceChildren();
    if (history.length === 0) return;
    host.append(el('h3', 'h6 mt-4 mb-2', 'Últimas evaluaciones'));
    const table = el('table', 'table table-sm align-middle small mb-0');
    const head = el('thead');
    const headRow = el('tr');
    for (const title of ['Hora', 'Probabilidad', 'Resultado', 'Pod']) headRow.append(el('th', 'fw-semibold', title));
    head.append(headRow);
    const body = el('tbody');
    for (const entry of history) {
      const row = el('tr');
      row.append(
        el('td', 'text-body-secondary', entry.at.toLocaleTimeString('es')),
        el('td', '', pct(entry.probability)),
        el('td', '', entry.premium === 1 ? 'Premium' : 'No premium'),
        el('td', 'font-monospace', entry.pod),
      );
      body.append(row);
    }
    table.append(head, body);
    host.append(table);
  }

  errorView(error) {
    const alert = el('div', 'alert alert-danger mb-0');
    alert.setAttribute('role', 'alert');
    let title = 'No se pudo evaluar la muestra';
    if (error.kind === 'network') title = 'No se pudo contactar con la API';
    else if (error.status === 503) title = 'El modelo todavía no está listo';
    else if (error.status === 422) title = 'La API rechazó los datos';
    else if (error.status === 502 || error.status === 504) title = 'La API no responde';
    alert.append(el('div', 'fw-semibold mb-1', title));
    if (error.detail) alert.append(el('div', 'small', error.detail));
    alert.append(el('div', 'small mt-2', 'Puedes volver a intentarlo en unos segundos.'));
    return alert;
  }

  successView({ result, modelInfo, warnings }) {
    const prediction = result.predictions[0];
    const premium = prediction.premium === 1;
    const probability = prediction.probability_premium;
    const threshold = modelInfo?.decision_threshold ?? null;

    const verdict = el('div', 'd-flex align-items-center gap-3 mb-3');
    verdict.append(
      el('span', `badge fs-5 px-3 py-2 ${premium ? 'text-bg-success' : 'text-bg-secondary'}`,
        premium ? 'Potencial premium' : 'No premium'),
    );

    const label = el('div', 'd-flex justify-content-between small mb-1');
    label.append(el('span', 'fw-semibold', 'Probabilidad de ser premium'), el('span', 'fw-semibold', pct(probability)));
    const track = el('div', 'prob-track');
    track.setAttribute('role', 'progressbar');
    track.setAttribute('aria-valuemin', '0');
    track.setAttribute('aria-valuemax', '100');
    track.setAttribute('aria-valuenow', String(Math.round(probability * 100)));
    const bar = el('div', `prob-bar ${premium ? 'bg-success' : 'bg-secondary'}`);
    bar.style.width = `${Math.max(probability * 100, 1)}%`;
    track.append(bar);
    if (threshold !== null) {
      const mark = el('div', 'prob-threshold');
      mark.style.left = `${threshold * 100}%`;
      mark.title = `Umbral de decisión: ${pct(threshold, 0)}`;
      track.append(mark);
    }

    const note = el('p', 'small text-body-secondary mt-2 mb-3');
    note.textContent = threshold === null
      ? premium ? 'El modelo clasifica la muestra como premium.' : 'El modelo no clasifica la muestra como premium.'
      : premium
        ? `La probabilidad supera el umbral de decisión (${pct(threshold, 0)}), por eso se marca como premium.`
        : `La probabilidad no alcanza el umbral de decisión (${pct(threshold, 0)}).`;

    const meta = el('div', 'd-flex flex-wrap gap-2 mb-3');
    meta.append(
      el('span', 'badge text-bg-light border', `${result.model_name} v${result.model_version}`),
      el('span', 'badge text-bg-light border', `alias ${result.model_alias}`),
      el('span', 'badge text-bg-info', `pod ${result.served_by}`),
      el('span', 'badge text-bg-light border font-monospace', `run ${result.run_id.slice(0, 8)}`),
    );

    const nodes = [verdict, label, track, note, meta];
    const warn = this.warningView(warnings);
    if (warn) nodes.push(warn);
    nodes.push(el('p', 'small text-body-secondary mb-0',
      'Estimación de apoyo entrenada con vinos tintos; no reemplaza la cata de expertos.'));
    return nodes;
  }

  warningView(warnings) {
    const names = FIELDS.filter((field) => warnings[field.name]);
    if (names.length === 0) return null;
    const outside = names.some((field) => warnings[field.name] === 'fuera');
    const alert = el('div', `alert ${outside ? 'alert-danger' : 'alert-warning'} py-2 small mb-3`);
    alert.setAttribute('role', 'alert');
    alert.append(el('div', 'fw-semibold mb-1',
      `${names.length} ${names.length === 1 ? 'variable atípica' : 'variables atípicas'} respecto a los tintos de entrenamiento`));
    alert.append(el('div', '', names.map((field) => field.label).join(' · ')));
    alert.append(el('div', 'mt-1',
      'El modelo aprendió con tintos: con valores así su estimación es menos fiable.'));
    return alert;
  }
}
