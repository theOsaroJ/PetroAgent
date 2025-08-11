import React, { useState } from "react";
import Papa from "papaparse";
import axios from "axios";

type Row = Record<string,string|number|null>;

export default function UploadPanel() {
  const [file, setFile] = useState<File|null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [model, setModel] = useState<"rf"|"xgb"|"gp"|"nn"|"transformer">("rf");
  const [saveDir, setSaveDir] = useState<string>("/app/outputs");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  function onPick(f?: File) {
    if (!f) return;
    setFile(f);
    // Ask backend to detect headers (robust for big files), but also parse quickly client-side for responsiveness
    Papa.parse(f, {
      header: true, preview: 5, dynamicTyping: true, complete: (r)=>{
        const cols = r.meta.fields || [];
        setColumns(cols);
        if (cols.length>1) {
          setFeatures(cols.slice(0, Math.max(1, cols.length-1)));
          setTarget(cols[cols.length-1]);
        }
      }
    });
  }

  async function detectRemoteColumns() {
    if (!file) return;
    const fd = new FormData(); fd.append("file", file);
    const { data } = await axios.post("/api/columns", fd, { headers: {"Content-Type":"multipart/form-data"} });
    if (data?.columns?.length) {
      setColumns(data.columns);
      if (!target) setTarget(data.columns[data.columns.length-1]);
      if (!features.length) setFeatures(data.columns.filter((c:string)=>c!==target));
    }
  }

  async function train() {
    if (!file || !features.length || !target) return;
    setBusy(true); setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("features", features.join(","));
      fd.append("target", target);
      fd.append("model_type", model);
      fd.append("save_dir", saveDir);
      const { data } = await axios.post("/api/upload", fd, { headers: {"Content-Type":"multipart/form-data"}});
      setResult(data);
    } catch (e:any) {
      alert("Training failed: " + (e?.response?.data?.detail || e.message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <input type="file" accept=".csv" onChange={e=>onPick(e.target.files?.[0]||null)} />
        <button onClick={detectRemoteColumns} className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600">Detect columns</button>
      </div>

      {!!columns.length && (
        <>
          <div>
            <label className="block text-sm mb-1">Features</label>
            <div className="flex flex-wrap gap-2">
              {columns.map(c=>(
                <label key={c} className="inline-flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={features.includes(c) && c!==target}
                    onChange={(e)=>{
                      if (c===target) return;
                      setFeatures(v=>e.target.checked ? [...v, c] : v.filter(x=>x!==c));
                    }}
                  />
                  <span className="text-sm">{c}</span>
                </label>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm mb-1">Target</label>
            <select value={target} onChange={e=>{ setTarget(e.target.value); setFeatures(f=>f.filter(x=>x!==e.target.value)); }}
              className="bg-slate-900 border border-white/10 rounded-lg p-2">
              <option value="" disabled>Select target</option>
              {columns.map(c=><option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm mb-1">Model</label>
          <select value={model} onChange={e=>setModel(e.target.value as any)} className="bg-slate-900 border border-white/10 rounded-lg p-2">
            <option value="rf">Random Forest</option>
            <option value="xgb">XGBoost</option>
            <option value="gp">Gaussian Process</option>
            <option value="nn">Neural Network (MLP)</option>
            <option value="transformer">Transformer (tabular)</option>
          </select>
        </div>
        <div>
          <label className="block text-sm mb-1">Save directory (in container)</label>
          <input value={saveDir} onChange={e=>setSaveDir(e.target.value)} className="w-full bg-slate-900 border border-white/10 rounded-lg p-2" />
        </div>
      </div>

      <div className="flex gap-2">
        <button disabled={!file || busy} onClick={train} className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-50">
          {busy ? "Training…" : "Train"}
        </button>
      </div>

      {result && (
        <div className="mt-4 space-y-2">
          <div className="text-sm text-slate-300">Model: <b>{result.model_type}</b></div>
          <pre className="bg-black/40 p-3 rounded-lg text-sm overflow-auto">{JSON.stringify(result.metrics, null, 2)}</pre>
          {result.artifacts?.parity_plot && (
            <div>
              <div className="text-sm text-slate-300 mb-2">Parity Plot</div>
              <img src={`/api/static?path=${encodeURIComponent(result.artifacts.parity_plot)}`} alt="parity" className="rounded-lg border border-white/10" />
              <p className="text-xs text-slate-500 mt-2">Saved: {result.artifacts.parity_plot}</p>
              <p className="text-xs text-slate-500">Model file: {result.artifacts.model_path}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
