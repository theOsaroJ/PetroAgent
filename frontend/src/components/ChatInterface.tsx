import React, { useState, useEffect } from "react";
import io from "socket.io-client";

const socket = io("/", { path: "/api/socket.io" });

interface Message {
  user?: string;
  bot?: string;
}

export default function ChatInterface() {
  const [msgs, setMsgs] = useState<Message[]>([]);
  const [input, setInput] = useState("");

  useEffect(() => {
    socket.on("chat response", (r: string) => {
      setMsgs((prev) => [...prev, { bot: r }]);
    });
    return () => {
      socket.off("chat response");
    };
  }, []);

  const send = () => {
    if (!input.trim()) return;
    setMsgs((prev) => [...prev, { user: input }]);
    socket.emit("chat request", input);
    setInput("");
  };

  return (
    <div className="mt-6">
      <div className="h-64 overflow-y-auto border p-2">
        {msgs.map((m, i) => (
          <div key={i} className={m.user ? "text-right" : "text-left"}>
            <span className="inline-block p-1 rounded bg-blue-100">
              {m.user ?? m.bot}
            </span>
          </div>
        ))}
      </div>
      <textarea
        className="w-full border p-2 mt-2"
        rows={3}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        placeholder="Type your prompt… (Shift+Enter = newline)"
      />
    </div>
  );
}
