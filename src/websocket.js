let latestScore = 0;
const listeners = [];

const ws = new WebSocket("ws://localhost:3000");
ws.onmessage = (e) => {
  const { anomalyScore } = JSON.parse(e.data);
  latestScore = anomalyScore;
  listeners.forEach((fn) => fn(latestScore));
};

export const getLatestScore = () => latestScore;
export const onScoreUpdate = (cb) => listeners.push(cb);
