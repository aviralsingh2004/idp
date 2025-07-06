import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:5000/api';

export function predictAero(inputs) {
  return axios.post(`${BASE_URL}/predict`, inputs).then(res => res.data);
}

export function getFeatureImportance() {
  return axios.get(`${BASE_URL}/feature-importance`).then(res => res.data);
}

export function getTelemetryComparison() {
  return axios.get(`${BASE_URL}/telemetry-comparison`).then(res => res.data);
}
