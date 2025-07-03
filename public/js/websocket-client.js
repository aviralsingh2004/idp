// public/js/websocket-client.js
export let latestScore = 0;

const ws = new WebSocket(`ws://${window.location.host}`);
ws.addEventListener('open',  () => console.log('WebSocket opened'));
ws.addEventListener('close', () => console.log('WebSocket closed'));
ws.addEventListener('message', ({ data }) => {
  const { anomalyScore } = JSON.parse(data);
  latestScore = anomalyScore;
  document.getElementById('score').textContent = anomalyScore.toFixed(3);
});
