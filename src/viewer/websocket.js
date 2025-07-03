// src/viewer/websocket.js
import { useEffect, useState } from 'react';

export function useWebSocket() {
  const [score, setScore] = useState(0);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:3000`);
    ws.onmessage = (e) => {
      const { anomalyScore } = JSON.parse(e.data);
      setScore(anomalyScore);
    };
    return () => ws.close();
  }, []);

  return score;
}
