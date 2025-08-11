import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

export default function Chat() {
  const [msg, setMsg] = useState("");
  const [items, setItems] = useState<{role:"user"|"assistant", text:string}[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:"smooth"}); }, [items]);

  async function send() {
    const text = msg.trim();
    if(!text) return;
    setItems(prev=>[...prev, {role:"user", text}]);
    setMsg("");
    try {
      const { data } = await axios.post("/api/chat", { message: text });
      setItems(prev=>[...prev, {role:"assistant", text: data.reply}]);
    } catch (e:any) {
      setItems(prev=>[...prev, {role:"assistant", text: "Error contacting model."}]);
    }
  }

  return (
    <div className="flex flex-col h-[520px]">
      <div className="flex-1 overflow-auto space-y-3 pr-1">
        {items.map((m,i)=>(
          <div key={i} className={m.role==="user"?"text-right":""}>
            <div className={`inline-block px-3 py-2 rounded-xl ${m.role==="user"?"bg-brand-600":"bg-slate-800"}`}>
              <pre className="whitespace-pre-wrap font-sans">{m.text}</pre>
            </div>
          </div>
        ))}
        <div ref={endRef}/>
      </div>
      <div className="mt-3 flex gap-2">
        <input
          value={msg}
          onChange={e=>setMsg(e.target.value)}
          onKeyDown={e=>{ if(e.key==="Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Ask PetroAgent… (Enter to send)"
          className="flex-1 bg-slate-900 border border-white/10 rounded-xl px-3 py-3 focus:outline-none"
        />
        <button onClick={send} className="px-4 rounded-xl bg-brand-600 hover:bg-brand-500">Send</button>
      </div>
    </div>
  );
}
