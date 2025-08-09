import React, { useState, useRef, useEffect } from "react";
import { chat } from "../lib/api";
import { Send, Bot, User } from "lucide-react";

type Msg = { role: "user"|"assistant"; text: string };

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([
    { role:"assistant", text:"Hi! I'm PetroAgent. Ask me about petroleum engineering or tell me to train a model on your CSV."}
  ]);
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:"smooth"}) },[messages]);

  async function send() {
    const text = input.trim();
    if(!text) return;
    setMessages(m=>[...m, {role:"user", text}]);
    setInput("");
    try {
      const reply = await chat(text);
      setMessages(m=>[...m, {role:"assistant", text: reply}]);
    } catch (e:any) {
      setMessages(m=>[...m, {role:"assistant", text:"(Error talking to model)"}]);
    }
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if(e.key==="Enter" && !e.shiftKey){
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="glass p-4 h-full flex flex-col">
      <div className="flex items-center gap-3 mb-3">
        <img src="/logo.png" className="w-10 h-10 rounded-lg" />
        <div>
          <div className="text-petro-text font-semibold text-lg">PetroAgent</div>
          <div className="text-petro-muted text-xs">LLM Assistant</div>
        </div>
      </div>

      <div className="flex-1 overflow-auto space-y-3 pr-1">
        {messages.map((m, i)=>(
          <div key={i} className={`flex items-start gap-2 ${m.role==="assistant"?"":"justify-end"}`}>
            {m.role==="assistant" && <Bot className="text-petro-accent w-5 h-5 mt-1"/>}
            <div className={`px-3 py-2 rounded-2xl max-w-[80%] ${m.role==="assistant"?"bg-white/5 text-petro-text":"bg-petro-accent text-slate-900"}`}>
              {m.text}
            </div>
            {m.role==="user" && <User className="text-petro-accent w-5 h-5 mt-1"/>}
          </div>
        ))}
        <div ref={endRef}></div>
      </div>

      <div className="mt-3 flex gap-2">
        <textarea
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask anything. Press Enter to send (Shift+Enter for newline)…"
          rows={2}
          className="input"
        />
        <button className="btn" onClick={()=>void send()} title="Send">
          <Send className="w-5 h-5"/>
        </button>
      </div>
    </div>
  )
}
