import React, { useState } from "react";
import { uploadCSV, trainModel, TrainPayload } from "../lib/api";
import { Database, PlayCircle } from "lucide-react";

export default function UploadTrain({ onArtifacts }: { onArtifacts: (paths: string[])=>void }) {
  const [schema, setSchema] = useState<null | Awaited<ReturnType<typeof uploadCSV>>>(null);
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState<string[]>([]);
  const [task, setTask] = useState<"regression"|"classification">("regression");
  const [modelType, setModelType] = useState<"rf"|"gp"|"xgb"|"nn"|"transformer">("xgb");
  const [outputDir, setOutputDir] = useState("run1");
  const [status, setStatus] = useState("");

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if(!f) return;
    setStatus("Uploading…");
    try {
      const s = await uploadCSV(f);
      setSchema(s);
      setStatus(`Uploaded ${s.n_rows} rows × ${s.n_cols} cols. Choose target & features.`);
    } catch(e:any) {
      setStatus("Upload failed.");
    }
  }

  function toggleFeature(c: string) {
    setFeatures(prev => prev.includes(c) ? prev.filter(x=>x!==c) : [...prev, c]);
  }

  async function train() {
    if(!schema || !target || features.length===0) { setStatus("Select target and at least one feature."); return; }
    const payload: TrainPayload = {
      upload_id: schema.upload_id,
      target, features, task,
      model_type: modelType,
      output_dir: outputDir || "run1",
      standardize: true,
      max_rows: modelType==="gp" ? 5000 : null
    };
    setStatus("Training… this can take a bit.");
    try {
      const res = await trainModel(payload);
      setStatus(`Done. Metrics: ${Object.entries(res.metrics).map(([k,v])=>`${k}=${v.toFixed(4)}`).join(" | ")}. Model saved at ${res.model_path}`);
      onArtifacts(res.artifacts);
    } catch(e:any) {
      setStatus("Training failed.");
    }
  }

  return (
    <div className="glass p-4 h-full flex flex-col">
      <div className="flex items-center gap-3 mb-3">
        <Database className="text-petro-accent"/>
        <div>
          <div className="text-petro-text font-semibold text-lg">Data & Training</div>
          <div className="text-petro-muted text-xs">Upload CSV → pick target & features → choose model → train</div>
        </div>
      </div>

      <div className="space-y-3">
        <input type="file" accept=".csv" onChange={onUpload} className="text-petro-text"/>
        {schema && (
          <div className="space-y-2">
            <div className="badge">Rows: {schema.n_rows} · Cols: {schema.n_cols}</div>

            <div>
              <div className="text-petro-muted text-sm mb-1">Target column</div>
              <select className="input" value={target} onChange={e=>setTarget(e.target.value)}>
                <option value="">Select target…</option>
                {schema.columns.map(c=><option key={c} value={c}>{c} ({schema.dtypes[c]})</option>)}
              </select>
            </div>

            <div>
              <div className="text-petro-muted text-sm mb-1">Feature columns</div>
              <div className="grid grid-cols-2 gap-2 max-h-48 overflow-auto">
                {schema.columns.map(c=>(
                  <label key={c} className="flex items-center gap-2 text-petro-text">
                    <input type="checkbox" checked={features.includes(c)} onChange={()=>toggleFeature(c)}/>
                    <span>{c} <span className="text-petro-muted text-xs">({schema.dtypes[c]})</span></span>
                  </label>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="text-petro-muted text-sm mb-1">Task</div>
                <select className="input" value={task} onChange={e=>setTask(e.target.value as any)}>
                  <option value="regression">Regression</option>
                  <option value="classification">Classification</option>
                </select>
              </div>
              <div>
                <div className="text-petro-muted text-sm mb-1">Model family</div>
                <select className="input" value={modelType} onChange={e=>setModelType(e.target.value as any)}>
                  <option value="xgb">XGBoost</option>
                  <option value="rf">Random Forest</option>
                  <option value="gp">Gaussian Process</option>
                  <option value="nn">Neural Network (MLP)</option>
                  <option value="transformer">TabTransformer</option>
                </select>
              </div>
            </div>

            <div>
              <div className="text-petro-muted text-sm mb-1">Save outputs under (inside container or relative):</div>
              <input className="input" value={outputDir} onChange={e=>setOutputDir(e.target.value)} placeholder="/app/data/outputs/run1"/>
              <div className="text-petro-muted text-xs mt-1">Default base is /app/data/outputs (mounted to host at ml_service/data/outputs)</div>
            </div>

            <button className="btn flex items-center gap-2" onClick={()=>void train()}>
              <PlayCircle className="w-5 h-5"/> Train
            </button>
          </div>
        )}

        <div className="text-petro-muted text-sm min-h-[1.5rem]">{status}</div>
      </div>
    </div>
  )
}
