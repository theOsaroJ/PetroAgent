import React from "react";
import Chat from "./Chat";
import Upload from "./Upload";
import ModelSelector from "./ModelSelector";
import Analysis from "./Analysis";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col items-center p-4">
      <img src="/banner.png" alt="Banner" className="w-full mb-4" />
      <div className="w-full max-w-3xl bg-white rounded-xl p-6 shadow-lg">
        <Chat />
        <Upload />
        <ModelSelector />
        <Analysis />
      </div>
    </div>
  );
}
