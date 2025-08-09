import React, { useEffect, useState } from "react";
import Chat from "./components/Chat";
import UploadTrain from "./components/UploadTrain";
import PlotGallery from "./components/PlotGallery";
import { listModels, describeData } from "./api";

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [fileInfo, setFileInfo] = useState<{file_id?: string, path?: string} | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [plots, setPlots] = useState<{[k:string]: string}>({});

  useEffect(() => {
    listModels().then(setModels).catch(()=>setModels([]));
  }, []);

  useEffect(() => {
    if (fileInfo?.file_id || fileInfo?.path) {
      describeData(fileInfo).then(d => setColumns(d.columns ?? [])).catch(()=>setColumns([]));
    }
  }, [fileInfo])

  return (
    <div className="min-h-screen">
      <header className="relative overflow-hidden">
        <img src="/banner.png" alt="banner" className="w-full h-64 object-cover opacity-95"/>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-4 bg-white/70 px-6 py-3 rounded-2xl shadow-lg">
            <img src="/logo.png" alt="logo" className="h-14 w-14"/>
            <h1 className="text-3xl font-extrabold tracking-tight text-brand-800">PetroAgent</h1>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 grid md:grid-cols-2 gap-8">
        <section className="md:col-span-2">
          <Chat />
        </section>

        <section>
          <UploadTrain
            models={models}
            columns={columns}
            onUploaded={setFileInfo}
            onPlots={(imgs)=>setPlots(imgs)}
          />
        </section>

        <section>
          <PlotGallery images={plots}/>
        </section>
      </main>

      <footer className="py-6 text-center text-sm text-slate-500">
        © {new Date().getFullYear()} PetroAgent — ML + Chat for Petroleum Engineering
      </footer>
    </div>
  );
}
