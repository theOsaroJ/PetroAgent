import React, { useState, useRef, useEffect } from "react";
import { sendChat } from "../api";

type Msg = { role: "user" | "assistant", content: string };

export default function Chat() {
  const [history, setHistory] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => { boxRef.current?.scrollTo({ top: boxRef.current.scrollHeight, behavior: "smooth" }); }, [history]);

  const submit = async () => {
    const content = text.trim();
    if (!content) return;
    const now = [...history, {role:"user", content}];
    setHistory(now);
    setText("");
    try {
      const reply = await sendChat(content, now);
      setHistory(prev => [...prev, {role:"assistant", content: reply}]);
    } catch (e:any) {
      setHistory(prev => [...prev, {role:"assistant", content: "Error: "+ (e?.message || e)}]);
    }
  }

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); submit();
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-4">
      <div ref={boxRef} className="h-72 overflow-y-auto space-y-3">
        {history.length === 0 && (
          <div className="text-slate-500 text-sm">Ask anything about petroleum engineering, or say “help me train a model”.</div>
        )}
        {history.map((m, i) => (
          <div key={i} className={`max-w-[85%] px-4 py-2 rounded-2xl ${m.role==='user'?'bg-brand-100 ml-auto':'bg-slate-100'}`}>
            <div className="text-sm whitespace-pre-wrap">{m.content}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-2">
        <input
          className="flex-1 border rounded-xl px-3 py-2 outline-brand-500"
          placeholder="Type your message and press Enter"
          value={text}
          onChange={e=>setText(e.target.value)}
          onKeyDown={onKey}
        />
        <button onClick={submit} className="bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-xl">Send</button>
      </div>
    </div>
  );
}
