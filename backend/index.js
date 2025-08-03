const fs = require('fs');
const express = require('express');
const cors = require('cors');
const { createServer } = require('http');
const { Server } = require('socket.io');
const routes = require('./routes');

// Load OpenAI key
const OPENAI_API_KEY = fs.readFileSync('api_key.txt', 'utf-8').trim();
if (!OPENAI_API_KEY) throw new Error('api_key.txt must contain your OpenAI key');
global.OPENAI_API_KEY = OPENAI_API_KEY;

const app = express();
app.use(cors());
app.use(express.json());
app.use('/api', routes);

const httpServer = createServer(app);
const io = new Server(httpServer, { cors: { origin: '*' } });

io.on('connection', socket => {
  socket.on('chat message', async msg => {
    const { sendMessage } = require('./services/chatService');
    const reply = await sendMessage(msg);
    socket.emit('chat response', reply);
  });
});

httpServer.listen(5000, () => console.log('Backend listening on 5000'));
