import { useEffect, useRef, useState } from "react";
import { chatLLM } from "../api";
import Loader from "./Loader";

export default function Chat({ datasetContext }) {
  const [messages, setMessages] = useState([
    { role: "system", content: "Welcome to PetroAgent. Ask me anything about petroleum data or training models." }
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef();

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function send(e) {
    e?.preventDefault?.();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    const ctx = datasetContext?.summary || "";
    setBusy(true);
    setMessages(m => [...m, { role: "user", content: text }]);
    try {
      const res = await chatLLM(text, ctx);
      setMessages(m => [...m, { role: "assistant", content: res.reply }]);
    } catch (err) {
      setMessages(m => [...m, { role: "assistant", content: "⚠️ Chat failed: " + (err?.message || "Unknown error") }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col h-[560px]">
      <div ref={scroller} className="flex-1 overflow-y-auto rounded-xl border border-slate-200 p-4 bg-white">
        {messages.filter(m=>m.role!=="system").map((m, i) => (
          <div key={i} className={`mb-3 ${m.role==="user" ? "text-right" : ""}`}>
            <div className={`inline-block px-4 py-2 rounded-2xl max-w-[85%] ${m.role==="user" ? "bg-brand-600 text-white" : "bg-slate-100"}`}>
              {m.content}
            </div>
          </div>
        ))}
        {busy && <Loader text="Thinking..." />}
      </div>
      <form onSubmit={send} className="mt-3 flex gap-2">
        <input
          value={input}
          onChange={(e)=>setInput(e.target.value)}
          onKeyDown={(e)=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send(); } }}
          placeholder="Ask PetroAgent… (press Enter to send)"
          className="flex-1 rounded-xl border border-slate-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-brand-400"
        />
        <button
          type="submit"
          className="px-4 py-3 rounded-xl bg-brand-600 text-white hover:bg-brand-700">
          Send
        </button>
      </form>
    </div>
  );
}
