import React, { useState } from "react";

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");

  const send = async () => {
    if (!input.trim()) return;
    const userMsg = input;
    setMessages(msgs => [...msgs, { sender: "user", text: userMsg }]);
    setInput("");

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMsg })
    });
    const data = await res.json();
    setMessages(msgs => [...msgs, { sender: "bot", text: data.response }]);
  };

  return (
    <>
      <div className="h-64 overflow-y-scroll border p-2 mb-2">
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.sender === "user" ? "text-right" : "text-left"}
          >
            <span className="inline-block bg-gray-200 p-1 rounded">
              {m.text}
            </span>
          </div>
        ))}
      </div>

      <input
        type="text"
        className="border p-2 w-full"
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === "Enter" && send()}
        placeholder="Type your message and press Enter"
      />
    </>
  );
}
