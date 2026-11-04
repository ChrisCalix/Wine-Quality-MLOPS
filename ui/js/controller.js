// Controlador (MVC): conecta los eventos de la vista con el modelo y el adaptador de la API.
import { toRecord } from './model.js';

const RETRY_MS = 10000;

export class Controller {
  constructor(model, view, api) {
    this.model = model;
    this.view = view;
    this.api = api;
    this.retryTimer = null;
  }

  init() {
    this.view.buildForm();
    this.model.subscribe((state) => this.view.render(state));
    this.view.onInput((name, value) => this.model.setValue(name, value));
    this.view.onPreset((key) => this.model.applyPreset(key));
    this.view.onClear(() => this.model.clear());
    this.view.onSubmit(() => this.evaluate());
    this.view.render(this.model.state);
    this.loadModelInfo();
  }

  async evaluate() {
    const errors = this.model.validate();
    if (Object.keys(errors).length > 0) {
      this.view.focusFirstError(errors);
      return;
    }
    this.model.setLoading();
    try {
      this.model.setResult(await this.api.predict(toRecord(this.model.state.values)));
    } catch (error) {
      this.model.setError(error);
    }
  }

  async loadModelInfo() {
    try {
      this.model.setModelInfo(await this.api.modelInfo());
      clearTimeout(this.retryTimer);
    } catch {
      this.model.setApiDown();
      // La API puede estar arrancando o cargando el modelo: se reintenta hasta que responda.
      this.retryTimer = setTimeout(() => this.loadModelInfo(), RETRY_MS);
    }
  }
}
