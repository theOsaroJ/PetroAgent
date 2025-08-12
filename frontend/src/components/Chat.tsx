import React, { useState } from 'react'
import { chat } from '../api'

type Msg = { role: 'user' | 'assistant', text: string }

export default function Chat() {
  const [input, setInput] = useState('Hello! How can I assist you with petroleum data today?')
  const [msgs, setMsgs] = useState<Msg[]>([
    { role: 'assistant', text: "Hello! I'm here and ready to assist you with any petroleum-related data or questions you have. How can I help you today?"}
  ])
  const [busy, setBusy] = useState(false)

  async function send() {
    if (!input.trim()) return
    const text = input.trim()
    setInput('')
    setMsgs(m => [...m, { role: 'user', text }])
    setBusy(true)
    try {
      const reply = await chat(text)
      setMsgs(m => [...m, { role: 'assistant', text: reply }])
    } catch (e:any) {
      setMsgs(m => [...m, { role: 'assistant', text: 'Error contacting model.' }])
      console.error(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid grid-rows-[1fr,auto] gap-4 lg:grid-cols-2 lg:gap-8">
      <div className="border border-slate-800 rounded-2xl p-4 h-[60vh] overflow-auto">
        <div className="space-y-4">
          {msgs.map((m, i) => (
            <div key={i} className={"max-w-[80%] rounded-2xl px-3 py-2 " + (m.role==='user'?'ml-auto bg-emerald-700':'bg-slate-800')}>
              {m.text}
            </div>
          ))}
        </div>
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>{ if(e.key==='Enter') send() }}
          className="flex-1 rounded-xl bg-slate-800 px-3 py-2 outline-none"
          placeholder="Ask PetroAgent..."
        />
        <button onClick={send} disabled={busy} className="px-4 py-2 rounded-xl bg-emerald-600 disabled:opacity-50">
          {busy?'Sending...':'Send'}
        </button>
      </div>
    </div>
  )
}
