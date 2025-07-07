import axios from 'axios'

// existing
export const predictAero           = params => axios.post('/api/predict', params).then(r => r.data)
export const getFeatureImportance  = ()     => axios.get('/api/feature-importance').then(r => r.data)
export const getTelemetryComparison= ()     => axios.get('/api/telemetry-comparison').then(r => r.data)

// new

export const startSimulation       = ()   => axios.post('/api/start-sim').then(r => r.data)
export const fetchLatestMetrics    = ()   => axios.get('/api/metrics').then(r => r.data)
export const fetchLapHistory       = lap  => axios.get(`/api/metrics/history?lap=${lap}`)
                                           .then(r => r.data)
export function fetchRawTelemetry({ year=2023, gp='Italian Grand Prix', session='Race', lap=1 }) {
  return axios
    .get('/api/raw-telemetry', { params: { year, gp, session, lap } })
    .then(res => res.data)
}
