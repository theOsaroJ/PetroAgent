import React, { useState } from "react";
import axios from "axios";

export default function Chat({ schema, selection, modelType }) {
  const [msg, setMsg] = useState("");
  const [conv, setConv] = useState([]);

  const send = async () => {
    if (!msg.trim()) return;
    setConv(c => [...c, { from: "user", text: msg }]);

    // include model + schema + selection in payload if you want:
    const payload = {
      prompt: msg,
      model_type: modelType,
      features: selection.features,
      target: selection.target
    };

    const { data } = await axios.post("/chat", payload);
    setConv(c => [...c, { from: "bot", text: data.response }]);
    setMsg("");
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto border rounded p-2 mb-2">
        {conv.map((m,i) => (
          <div
            key={i}
            className={`my-1 p-1 ${m.from==="user"?"text-right":"text-left"}`}
          >
            {m.text}
          </div>
        ))}
      </div>
      <input
        className="w-full p-2 border rounded"
        value={msg}
        onChange={e => setMsg(e.target.value)}
        onKeyDown={e => e.key === "Enter" && send()}
        placeholder="Type your message and press Enter"
      />
    </div>
  );
}
