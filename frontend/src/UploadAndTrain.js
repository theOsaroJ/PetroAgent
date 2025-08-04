import React, { useState } from "react";

export default function UploadAndTrain() {
  const [cols,setCols] = useState([]);
  const [fileId,setFileId] = useState("");
  const [features,setFeatures] = useState([]);
  const [target,setTarget] = useState("");
  const [model,setModel] = useState("neural_network");
  const [savePath,setSavePath] = useState("models");
  const [result,setResult] = useState(null);

  const onUpload = async e => {
    const data = new FormData(); data.append("file", e.target.files[0]);
    const res = await fetch("/api/upload",{ method:"POST", body:data });
    const { id, columns } = await res.json();
    setFileId(id); setCols(columns);
  };

  const train = async () => {
    const res = await fetch("/api/train", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        id: fileId, features, target, model_type:model, save_path: savePath
      })
    });
    setResult(await res.json());
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow space-y-4">
      <h2 className="text-2xl font-semibold">Upload & Train</h2>
      <input type="file" onChange={onUpload} className="block"/>
      {cols.length>0 && <>
        <div>
          <label>Features:</label>
          <select multiple className="border p-2" size={5}
            onChange={e=>setFeatures([...e.target.selectedOptions].map(o=>o.value))}>
            {cols.map(c=><option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label>Target:</label>
          <select className="border p-2"
            onChange={e=>setTarget(e.target.value)}>
            <option/> {cols.map(c=><option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label>Model:</label>
          <select value={model} onChange={e=>setModel(e.target.value)}>
            <option value="neural_network">Neural Network</option>
            <option value="random_forest">Random Forest</option>
            <option value="gaussian_process">Gaussian Process</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>
        <div>
          <label>Save Path:</label>
          <input type="text" value={savePath}
            onChange={e=>setSavePath(e.target.value)} className="border p-2 w-full"/>
        </div>
        <button onClick={train}
          className="bg-indigo-600 text-white px-4 py-2 rounded shadow">
          Train
        </button>
      </>}
      {result && <>
        <p className="font-medium">MSE: {result.mse.toFixed(4)}</p>
        <img src={result.plot_b64} alt="fit plot" className="border"/>
        <p>Model saved at: <code>{result.model_file}</code></p>
      </>}
    </div>
  );
}
