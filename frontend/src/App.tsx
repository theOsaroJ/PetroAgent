import React, { useState } from "react";
import Chat from "./components/Chat";
import UploadTrain from "./components/UploadTrain";
import PlotGallery from "./components/PlotGallery";
import { Cpu, LineChart } from "lucide-react";

export default function App() {
  const [arts, setArts] = useState<string[]>([]);
  return (
    <div className="min-h-screen text-petro-text" style={{backgroundImage:'url(/banner.png)', backgroundSize:'cover', backgroundPosition:'center'}}>
      <div className="min-h-screen backdrop-blur-sm bg-gradient-to-b from-[#0b1220]/80 to-[#0b1220]/95">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <header className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <img src="/logo.png" className="w-10 h-10 rounded-lg"/>
              <div className="font-bold text-xl">PetroAgent</div>
              <span className="badge"><Cpu className="w-4 h-4"/> GPT + ML</span>
              <span className="badge"><LineChart className="w-4 h-4"/> EDA & Plots</span>
            </div>
            <a className="text-petro-muted hover:text-petro-accent transition text-sm" href="https://example.com" target="_blank">Docs</a>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-[78vh]">
            <div className="lg:col-span-2 grid grid-rows-2 gap-4">
              <UploadTrain onArtifacts={(p)=>setArts(p)}/>
              <PlotGallery paths={arts}/>
            </div>
            <Chat/>
          </div>

          <footer className="mt-6 text-center text-petro-muted text-xs">
            © 2025 PetroAgent. Built for high-impact petroleum analytics.
          </footer>
        </div>
      </div>
    </div>
  )
}
