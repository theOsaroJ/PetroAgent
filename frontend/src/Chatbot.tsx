import React, { useState, useRef } from 'react';
import axios from 'axios';

export default function Chatbot() {
  const [messages, setMessages] = useState<{ from:string; text:string }[]>([]);
  const [input, setInput] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);

  const send = async () => {
    if (!input.trim()) return;
    const userMsg = { from:'user', text: input };
    setMessages(ms => [...ms, userMsg]);
    setInput('');
    const res = await axios.post('http://localhost:7000/chat', { message: input });
    setMessages(ms => [...ms, { from:'bot', text: res.data.reply }]);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto mb-2">
        {messages.map((m,i) => (
          <div key={i} className={`my-1 ${m.from==='user' ? 'text-right' : 'text-left'}`}>
            <span className="inline-block p-2 rounded bg-gray-200">{m.text}</span>
          </div>
        ))}
      </div>
      <textarea
        ref={ref}
        className="border p-2"
        rows={3}
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => {
          if (e.key==='Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
      />
      <button onClick={send} className="mt-2 p-2 bg-blue-500 text-white rounded">Send</button>
    </div>
  );
}
