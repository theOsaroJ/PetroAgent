import React, { useState } from "react";
import ChatInterface from "./components/ChatInterface";
import FileUploader   from "./components/FileUploader";
import ModelSelector  from "./components/ModelSelector";

export default function App() {
  const [columns, setColumns]           = useState([]);
  const [tableData, setTableData]       = useState([]);
  const [selectedInputs, setSelectedInputs] = useState([]);
  const [target, setTarget]             = useState("");
  const [modelType, setModelType]       = useState("neural");

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">PetroAgent</h1>

      <FileUploader
        onDetect={cols => setColumns(cols)}
        setTableData={setTableData}
      />

      {columns.length > 0 && (
        <div className="mb-4">
          <div className="font-semibold">Inputs:</div>
          {columns.map(col => (
            <label key={col} className="block">
              <input
                type="checkbox"
                value={col}
                onChange={e =>
                  e.target.checked
                    ? setSelectedInputs([...selectedInputs, col])
                    : setSelectedInputs(selectedInputs.filter(c => c !== col))
                }
              />
              {col}
            </label>
          ))}

          <div className="font-semibold mt-2">Target:</div>
          <select onChange={e => setTarget(e.target.value)} value={target}>
            <option value="">-- select --</option>
            {columns.map(col => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>
      )}

      <ModelSelector modelType={modelType} setModelType={setModelType} />
      <ChatInterface />
    </div>
  );
}
