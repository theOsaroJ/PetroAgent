import { useState } from "react";
import Chat from "./components/Chat.jsx";
import UploadAndTrain from "./components/UploadAndTrain.jsx";

export default function App() {
  const [datasetContext, setDatasetContext] = useState(null);

  return (
    <div>
      <header className="bg-white/70 backdrop-blur sticky top-0 z-10 border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-3">
          <img src="/logo.png" className="h-10 w-10" alt="logo"/>
          <div>
            <div className="text-xl font-semibold">PetroAgent</div>
            <div className="text-xs text-slate-500">Chat • Analyze • Train • Deploy</div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <section className="grid lg:grid-cols-2 gap-8 items-start">
          <div className="order-2 lg:order-1">
            <UploadAndTrain onContext={setDatasetContext} />
          </div>
          <div className="order-1 lg:order-2">
            <div className="mb-6">
              <img src="/banner.png" alt="banner" className="rounded-2xl shadow-lg"/>
            </div>
            <Chat datasetContext={datasetContext} />
          </div>
        </section>
      </main>

      <footer className="py-6 text-center text-slate-500 text-sm">
        © PetroAgent — Built for petroleum data workflows.
      </footer>
    </div>
  );
}
