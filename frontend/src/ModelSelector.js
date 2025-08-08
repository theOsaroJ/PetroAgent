import React, { useState } from "react";
import axios from "axios";

export default function ModelSelector() {
  const [features,setFeatures]=useState(""),[target,setTarget]=useState(""),
        [model,setModel]=useState("nn"),[status,setStatus]=useState();
  const train=async()=>{
    // user must have uploaded file and picked CSV features manually...
    // for brevity, omitted UI for feature list
    const res=await axios.post("http://localhost:8000/train", {
      features: features.split(","), target, model_type: model
    });
    setStatus(res.data);
  };
  return (
    <div className="mb-4">
      <h2 className="font-semibold mb-1">Train Model</h2>
      <input className="border p-1 mr-2" placeholder="features (f1,f2)" value={features}
             onChange={e=>setFeatures(e.target.value)} />
      <input className="border p-1 mr-2" placeholder="target" value={target}
             onChange={e=>setTarget(e.target.value)} />
      <select className="border p-1 mr-2" value={model} onChange={e=>setModel(e.target.value)}>
        <option value="nn">Neural Net</option>
        <option value="rf">Random Forest</option>
        <option value="gp">Gaussian Process</option>
        <option value="xgb">XGBoost</option>
        <option value="tf">Transformer</option>
      </select>
      <button className="px-4 py-1 bg-green-500 text-white rounded" onClick={train}>
        Train
      </button>
      {status && <pre className="mt-2 bg-gray-100 p-2 rounded">{JSON.stringify(status,null,2)}</pre>}
    </div>
  );
}
