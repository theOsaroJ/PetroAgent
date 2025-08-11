import React from "react";
import Chat from "./components/Chat";
import UploadPanel from "./components/UploadPanel";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-hero">
        <div className="backdrop-blur-sm bg-black/40">
          <div className="max-w-6xl mx-auto px-6 py-8 flex items-center gap-4">
            <img src="/logo.png" className="w-12 h-12 rounded-lg" />
            <div>
              <h1 className="text-2xl font-bold">PetroAgent</h1>
              <p className="text-slate-300">Chat • Analyze • Train • Deploy</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto w-full px-6 py-8 grid md:grid-cols-2 gap-6">
        <section className="border border-white/10 rounded-2xl p-4 bg-white/5 shadow-xl">
          <h2 className="text-lg font-semibold mb-2">Chat</h2>
          <Chat />
        </section>
        <section className="border border-white/10 rounded-2xl p-4 bg-white/5 shadow-xl">
          <h2 className="text-lg font-semibold mb-2">Train Models</h2>
          <UploadPanel />
        </section>
      </main>

      <footer className="text-center text-slate-400 py-6">
        © {new Date().getFullYear()} PetroAgent
      </footer>
    </div>
  );
}
