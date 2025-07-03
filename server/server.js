// server/server.js
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

const server = createServer(app);
const wss = new WebSocketServer({ server });

// serve static wing.glb (optional if React frontend fetches directly)
app.use(express.static(path.join(__dirname, '..', 'public')));

wss.on('connection', (ws) => {
  console.log('✅ Client connected');
  const send = () => {
    if (ws.readyState === ws.OPEN) {
      const anomalyScore = Math.random(); // Simulated score
      ws.send(JSON.stringify({ anomalyScore }));
      setTimeout(send, 500);
    }
  };
  send();

  ws.on('close', () => console.log('❌ Client disconnected'));
});

server.listen(port, () => {
  console.log(`🚀 WebSocket server running at http://localhost:${port}`);
});
