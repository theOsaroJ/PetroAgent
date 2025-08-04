import React, { useState } from 'react';
import axios from 'axios';

export default function Chat() {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [features, setFeatures] = useState("");
  const [target, setTarget] = useState("");
  const [modelType, setModelType] = useState("random_forest");
  const [savePath, setSavePath] = useState("");

  const sendChat = async () => {
    if (!input) return;
    setMsgs(m=>[...m, {from:"user",text:input}]);
    const res = await axios.post("/api/chat", new URLSearchParams({ prompt: input }));
    setMsgs(m=>[...m, {from:"bot",text:res.data.reply}]);
    setInput("");
  };

  const uploadCSV = async () => {
    const form = new FormData();
    form.append("file", file);
    const res = await axios.post("/api/upload", form);
    setColumns(res.data.columns);
  };

  const trainModel = async () => {
    const res = await axios.post("/api/train", new URLSearchParams({
      model_type: modelType,
      features,
      target,
      save_path: savePath
    }));
    setMsgs(m=>[...m, {from:"bot",text:JSON.stringify(res.data)}]);
  };

  return (
    <div className="w-full max-w-2xl bg-white p-6 rounded-lg shadow-lg">
      <div className="mb-4">
        <input type="file" onChange={e=>setFile(e.target.files[0])}/>
        <button onClick={uploadCSV} className="ml-2 px-3 py-1 bg-blue-500 text-white rounded">Upload CSV</button>
      </div>
      {columns.length>0 && (
        <div className="mb-4 space-y-2">
          <div className="text-sm text-gray-600">Columns: {columns.join(", ")}</div>
          <input
            className="w-full border p-2 rounded"
            placeholder="features (comma separated)"
            value={features}
            onChange={e=>setFeatures(e.target.value)}
          />
          <input
            className="w-full border p-2 rounded"
            placeholder="target column"
            value={target}
            onChange={e=>setTarget(e.target.value)}
          />
          <select
            className="w-full border p-2 rounded"
            value={modelType}
            onChange={e=>setModelType(e.target.value)}
          >
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
            <option value="neural_net">Neural Network</option>
            <option value="transformer">Transformer</option>
          </select>
          <input
            className="w-full border p-2 rounded"
            placeholder="save path (e.g. /models/my.pkl)"
            value={savePath}
            onChange={e=>setSavePath(e.target.value)}
          />
          <button onClick={trainModel} className="w-full bg-green-500 text-white p-2 rounded">Train Model</button>
        </div>
      )}
      <div className="h-64 overflow-auto mb-4 border p-3 rounded">
        {msgs.map((m,i)=>(
          <div key={i} className={m.from==="user"?"text-right":"text-left"}>
            <span className={`inline-block p-2 rounded ${m.from==="user"?"bg-blue-100":"bg-gray-100"}`}>
              {m.text}
            </span>
          </div>
        ))}
      </div>
      <div className="flex">
        <input
          className="flex-grow border p-2 rounded"
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>e.key==="Enter"&&sendChat()}
          placeholder="Type a message…"
        />
        <button onClick={sendChat} className="ml-2 bg-blue-600 text-white p-2 rounded">Send</button>
      </div>
    </div>
  );
}
