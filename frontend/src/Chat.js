import React, { useState } from "react";
import axios from "axios";

export default function Chat() {
  const [msg, setMsg] = useState(""), [log, setLog] = useState([]);
  const send = async () => {
    const res = await axios.post("/api/chat", new URLSearchParams({ prompt: msg }));
    setLog([...log, { from: "you", text: msg }, { from: "bot", text: res.data.reply }]);
    setMsg("");
  };
  return (
    <div className="mb-4">
      <div className="h-48 overflow-auto p-2 border rounded mb-2">
        {log.map((m,i)=><div key={i} className={m.from==="you"?"text-right":"text-left"}>{m.text}</div>)}
      </div>
      <input
        className="w-full p-2 border rounded"
        value={msg}
        onChange={e=>setMsg(e.target.value)}
        onKeyDown={e=>e.key==="Enter" && send()}
        placeholder="Type your question..."
      />
    </div>
  );
}
