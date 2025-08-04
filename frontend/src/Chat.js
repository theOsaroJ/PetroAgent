import React, { useState, useRef } from "react";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const endRef = useRef();

  const send = async () => {
    if (!input) return;
    const user = { from: "user", text: input };
    setMessages(msgs => [...msgs, user]);
    setInput("");
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ message: input })
    });
    const { reply } = await res.json();
    setMessages(msgs => [...msgs, { from:"bot", text:reply }]);
    endRef.current?.scrollIntoView({ behavior:"smooth" });
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow">
      <h2 className="text-2xl font-semibold mb-4">Chat with PetroAgent</h2>
      <div className="h-64 overflow-y-auto mb-4 space-y-2">
        {messages.map((m,i)=>(
          <div key={i} className={m.from==="user"?"text-right":"text-left"}>
            <span className={`inline-block px-4 py-2 rounded ${
              m.from==="user"?"bg-blue-200":"bg-gray-200"
            }`}>
              {m.text}
            </span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <input
        className="w-full p-3 rounded border"
        placeholder="Type and hit Enter..."
        value={input}
        onChange={e=>setInput(e.target.value)}
        onKeyDown={e=>e.key==="Enter" && send()}
      />
    </div>
  );
}
