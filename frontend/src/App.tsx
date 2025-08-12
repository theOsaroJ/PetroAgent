import React, { useState } from 'react'
import Chat from './components/Chat'
import Trainer from './components/Trainer'

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat'|'train'>('chat')
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 grid place-content-center">⛽</div>
            <div className="text-xl font-semibold">PetroAgent</div>
          </div>
          <nav className="flex items-center gap-2">
            <button onClick={() => setActiveTab('chat')} className={"px-3 py-1 rounded-lg " + (activeTab==='chat'?'bg-emerald-600':'bg-slate-800')}>Chat</button>
            <button onClick={() => setActiveTab('train')} className={"px-3 py-1 rounded-lg " + (activeTab==='train'?'bg-emerald-600':'bg-slate-800')}>Train</button>
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 py-6">
        {activeTab === 'chat' ? <Chat/> : <Trainer/>}
      </main>
      <footer className="text-center text-slate-400 py-10">© 2025 PetroAgent</footer>
    </div>
  )
}
