import React, { useState } from "react";
import Chat from "./Chat";
import Upload from "./Upload";

export default function App() {
  const [columns, setColumns] = useState([]);
  const [selection, setSelection] = useState({ features: [], target: "" });
  const [modelType, setModelType] = useState("neural");

  return (
    <div className="min-h-screen flex flex-col">
      {/* header with logo */}
      <header className="flex items-center bg-blue-700 p-4 text-white">
        <img src="/images/logo.png" alt="Logo" className="h-10 mr-4"/>
        <h1 className="text-2xl font-bold">PetroAgent</h1>
      </header>

      {/* banner under header */}
      <img src="/images/banner.png" alt="Banner" className="w-full object-cover h-32"/>

      <div className="flex flex-1 overflow-hidden">
        {/* sidebar */}
        <aside className="w-1/4 p-4 bg-gray-100 overflow-auto">
          <Upload
            onSchema={cols => setColumns(cols)}
            onSelect={sel => setSelection(sel)}
          />
          <div className="mt-6">
            <label className="block mb-2 font-semibold">Select model:</label>
            <select
              value={modelType}
              onChange={e => setModelType(e.target.value)}
              className="w-full p-2 border rounded"
            >
              <option value="neural">Neural Network</option>
              <option value="gp">Gaussian Process</option>
              <option value="rf">Random Forest</option>
              <option value="transformer">Transformer</option>
            </select>
          </div>
        </aside>

        {/* chat area */}
        <main className="w-3/4 flex flex-col p-4">
          <Chat
            schema={columns}
            selection={selection}
            modelType={modelType}
          />
        </main>
      </div>
    </div>
  );
}
