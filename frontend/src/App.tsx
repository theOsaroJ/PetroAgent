import React, { useState } from 'react'
import Chat from './components/Chat'
import Trainer from './components/Trainer'

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat'|'train'>('chat')

  return (
    <div className="min-h-screen">
      {/* Top bar with your PNG logo */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src="/logo.png"
              width={32}
              height={32}
              alt="PetroAgent logo"
              className="w-8 h-8 rounded-xl object-contain"
            />
            <div className="text-xl font-semibold">PetroAgent</div>
            <div className="text-sm text-slate-400 hidden sm:block">Chat • Analyze • Train • Deploy</div>
          </div>
          <nav className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('chat')}
              className={
                "px-3 py-1 rounded-lg transition-colors " +
                (activeTab==='chat' ? 'bg-emerald-600' : 'bg-slate-800 hover:bg-slate-700')
              }
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab('train')}
              className={
                "px-3 py-1 rounded-lg transition-colors " +
                (activeTab==='train' ? 'bg-emerald-600' : 'bg-slate-800 hover:bg-slate-700')
              }
            >
              Train
            </button>
          </nav>
        </div>
      </header>

      {/* Banner hero using your PNG */}
      <section
        aria-label="Hero"
        className="w-full border-b border-slate-800"
      >
        <div className="max-w-6xl mx-auto px-4">
          <div className="relative overflow-hidden rounded-2xl my-4">
            <img
              src="/banner.png"
              alt="PetroAgent banner"
              className="w-full h-48 sm:h-56 md:h-64 object-cover"
            />
            <div className="absolute inset-0 bg-slate-900/30"></div>
            <div className="absolute bottom-4 left-4 right-4 flex items-center gap-3">
              <img
                src="/logo.png"
                alt="PetroAgent"
                className="w-10 h-10 rounded-xl object-contain border border-white/10"
              />
              <div>
                <div className="text-lg md:text-xl font-semibold">Your petroleum engineering copilot</div>
                <div className="text-slate-300 text-sm">Upload data • train models • chat with domain knowledge</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-6">
        {activeTab === 'chat' ? <Chat/> : <Trainer/>}
      </main>

      <footer className="text-center text-slate-400 py-10">© 2025 PetroAgent</footer>
    </div>
  )
}
