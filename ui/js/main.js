// Punto de entrada: crea el modelo, la vista y el cliente de la API, y los conecta.
import { ApiClient } from './apiClient.js';
import { Controller } from './controller.js';
import { WineModel } from './model.js';
import { View } from './view.js';

new Controller(new WineModel(), new View(document), new ApiClient('/api')).init();
