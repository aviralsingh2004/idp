// server/server.js
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.static(path.join(__dirname, '..', 'public')));

const py = spawn('python', ['infer.py'], { cwd: path.join(__dirname) });

wss.on('connection', (ws) => {
  const sendScore = (data) => {
    const lines = data.toString().split('\n');
    lines.forEach((line) => {
      if (line.trim() && ws.readyState === ws.OPEN) {
        ws.send(line);
      }
    });
  };

  py.stdout.on('data', sendScore);

  ws.on('close', () => {
    py.stdout.off('data', sendScore);
  });
});

server.listen(3000, () => console.log('✅ WebSocket server running on port 3000'));
