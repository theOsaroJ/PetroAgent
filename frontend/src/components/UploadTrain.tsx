import React, { useState } from "react";
import { uploadCSV, trainModel, generatePlots } from "../api";

export default function UploadTrain({
  models, columns, onUploaded, onPlots
}: {
  models: string[], columns: string[],
  onUploaded: (v: {file_id?: string, path?: string}) => void,
  onPlots: (imgs: Record<string,string>) => void
}) {
  const [file, setFile] = useState<File | null>(null);
  const [features, setFeatures] = useState("");
  const [target, setTarget] = useState("");
  const [model, setModel] = useState(models?.[0] || "NeuralNet");
  const [savePath, setSavePath] = useState("/app/artifacts/petro_model.pkl");
  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [fileInfo, setFileInfo] = useState<{file_id?: string, path?: string} | null>(null);

  const doUpload = async () => {
    if (!file) return alert("Choose a CSV file");
    const info = await uploadCSV(file);
    setFileInfo(info);
    onUploaded(info);
  };

  const doPlots = async () => {
    if (!fileInfo) return alert("Upload a CSV first");
    const imgs = await generatePlots({
      file_id: fileInfo.file_id, path: fileInfo.path,
      features: features.split(",").map(s=>s.trim()).filter(Boolean),
      target: target.trim(),
    });
    onPlots(imgs);
  };

  const doTrain = async () => {
    if (!fileInfo) return alert("Upload a CSV first");
    const feats = features.split(",").map(s=>s.trim()).filter(Boolean);
    const tgt = target.trim();
    if (!feats.length || !tgt) return alert("Provide features and target");
    setTraining(true); setMetrics(null);
    try {
      const res = await trainModel({
        file_id: fileInfo.file_id, path: fileInfo.path,
        features: feats, target: tgt, model, save_path: savePath
      });
      setMetrics(res.metrics);
      alert(`Saved model → ${res.save_path}`);
    } catch (e:any) {
      alert(`Training error: ${e?.message || e}`);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-4 space-y-4">
      <h2 className="text-xl font-semibold">Upload & Train</h2>

      <div className="flex items-center gap-3">
        <input type="file" accept=".csv" onChange={e=>setFile(e.target.files?.[0] || null)} />
        <button className="bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-xl"
          onClick={doUpload}>Upload</button>
      </div>

      <div className="grid sm:grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-slate-600">Features (comma-separated)</label>
          <input className="w-full border rounded-lg px-3 py-2" placeholder="GR, RHOB, NPHI, ..." value={features} onChange={e=>setFeatures(e.target.value)} />
          {!!columns.length && <div className="text-[11px] text-slate-500 mt-1">Columns: {columns.join(", ")}</div>}
        </div>
        <div>
          <label className="block text-sm text-slate-600">Target</label>
          <input className="w-full border rounded-lg px-3 py-2" placeholder="ROP" value={target} onChange={e=>setTarget(e.target.value)} />
        </div>
      </div>

      <div className="grid sm:grid-cols-3 gap-3 items-end">
        <div>
          <label className="block text-sm text-slate-600">Model</label>
          <select className="w-full border rounded-lg px-3 py-2" value={model} onChange={e=>setModel(e.target.value)}>
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div className="sm:col-span-2">
          <label className="block text-sm text-slate-600">Save to path</label>
          <input className="w-full border rounded-lg px-3 py-2" value={savePath} onChange={e=>setSavePath(e.target.value)} />
        </div>
      </div>

      <div className="flex gap-3">
        <button className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-xl disabled:opacity-60"
          disabled={training} onClick={doTrain}>{training ? "Training..." : "Train"}</button>
        <button className="bg-slate-700 hover:bg-slate-800 text-white px-4 py-2 rounded-xl"
          onClick={doPlots}>Generate Plots</button>
      </div>

      {metrics && (
        <div className="text-sm text-slate-700">
          <div className="font-semibold">Metrics</div>
          <pre className="bg-slate-50 border rounded-lg p-2">{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
