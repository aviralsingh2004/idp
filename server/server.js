// server/server.js
import path from 'path';
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// 1) Serve your front-end
app.use(express.static(path.join(__dirname, '..', 'public')));

// 2) Create HTTP + WebSocket servers
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// 3) On client connect, start sending anomaly scores
wss.on('connection', ws => {
  console.log('Client connected');
  const sendScore = () => {
    if (ws.readyState === ws.OPEN) {
      // TODO → replace Math.random() with your autoencoder inference
      const anomalyScore = Math.random();
      ws.send(JSON.stringify({ anomalyScore }));
      setTimeout(sendScore, 500);
    }
  };
  sendScore();

  ws.on('close', () => console.log('Client disconnected'));
});

// 4) Start listening
server.listen(port, () =>
  console.log(`✅ Server running at http://localhost:${port}`)
);
