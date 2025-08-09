import { useMemo, useState } from "react";
import { uploadCSV, trainModel } from "../api";
import FancyCard from "./FancyCard";
import Loader from "./Loader";

const MODEL_OPTS = [
  { key: "neural_net", label: "Neural Net (PyTorch MLP)" },
  { key: "tab_transformer", label: "Transformer (Tabular)" },
  { key: "random_forest", label: "Random Forest" },
  { key: "xgboost", label: "XGBoost" },
  { key: "gp", label: "Gaussian Process" }
];

export default function UploadAndTrain({ onContext }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState({ head: [], columns: [], file_rel_path: "" });
  const [features, setFeatures] = useState([]);
  const [target, setTarget] = useState("");
  const [model, setModel] = useState(MODEL_OPTS[0].key);
  const [savePath, setSavePath] = useState("artifacts/run1");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [notes, setNotes] = useState("");

  async function doUpload() {
    if (!file) return;
    setBusy(true);
    setResult(null);
    try {
      const res = await uploadCSV(file);
      setPreview(res);
      // build dataset context for LLM
      const summary = `Columns: ${res.columns.join(", ")}.`;
      onContext?.({ summary, file_rel_path: res.file_rel_path });
    } catch (e) {
      alert("Upload failed: " + (e?.response?.data?.detail || e.message));
    } finally {
      setBusy(false);
    }
  }

  async function doTrain() {
    if (!preview.file_rel_path || features.length === 0 || !target) {
      alert("Upload a CSV, choose at least one feature, and set a target.");
      return;
    }
    setBusy(true);
    setResult(null);
    try {
      const payload = {
        file_rel_path: preview.file_rel_path,
        features,
        target,
        model_type: model,
        save_dir: savePath,
        params: {}     // you can extend to include hyperparams from UI later
      };
      const res = await trainModel(payload);
      setResult(res);
      setNotes(res.notes || "");
    } catch (e) {
      alert("Train failed: " + (e?.response?.data?.detail || e.message));
    } finally {
      setBusy(false);
    }
  }

  const selectableColumns = useMemo(()=> preview.columns || [], [preview.columns]);

  return (
    <div className="grid gap-6">
      <FancyCard title="Upload CSV" subtitle="Preview first rows and expose columns to the assistant.">
        <div className="flex items-center gap-3">
          <input type="file" accept=".csv" onChange={(e)=>setFile(e.target.files?.[0]||null)} />
          <button onClick={doUpload} className="px-4 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-700">Upload</button>
          {busy && <Loader text="Uploading..." />}
        </div>

        {preview.head.length > 0 && (
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full text-sm border">
              <thead className="bg-slate-100">
                <tr>
                  {Object.keys(preview.head[0] || {}).map((k)=>(
                    <th key={k} className="px-2 py-1 border">{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.head.map((row,i)=>(
                  <tr key={i}>
                    {Object.keys(row).map((k)=>(
                      <td key={k} className="px-2 py-1 border">{String(row[k])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </FancyCard>

      <FancyCard title="Train Model" subtitle="Pick features, target and model family.">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-slate-600 mb-1">Features</label>
            <select multiple className="w-full border rounded-xl p-2 h-40"
                    value={features}
                    onChange={(e)=> setFeatures(Array.from(e.target.selectedOptions).map(o=>o.value))}>
              {selectableColumns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <div className="text-xs text-slate-500 mt-1">Hold Ctrl/Cmd to select multiple.</div>
          </div>

          <div className="grid gap-4">
            <div>
              <label className="block text-sm text-slate-600 mb-1">Target</label>
              <select className="w-full border rounded-xl p-2"
                      value={target}
                      onChange={(e)=>setTarget(e.target.value)}>
                <option value="">Select target…</option>
                {selectableColumns.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-600 mb-1">Model</label>
              <select className="w-full border rounded-xl p-2"
                      value={model}
                      onChange={(e)=>setModel(e.target.value)}>
                {MODEL_OPTS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-600 mb-1">Save artifacts under</label>
              <input value={savePath} onChange={(e)=>setSavePath(e.target.value)}
                     className="w-full border rounded-xl p-2"
                     placeholder="artifacts/run1" />
            </div>

            <div className="flex gap-2">
              <button onClick={doTrain} className="px-4 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700">Train</button>
              {busy && <Loader text="Training..." />}
            </div>
          </div>
        </div>

        {result && (
          <div className="mt-6 grid gap-4">
            <div>
              <h3 className="font-semibold">Metrics</h3>
              <pre className="bg-slate-50 rounded-xl p-3 text-sm overflow-x-auto">{JSON.stringify(result.metrics, null, 2)}</pre>
            </div>
            <div>
              <h3 className="font-semibold">Artifacts</h3>
              <ul className="list-disc ml-6">
                {result.artifacts.map((a,i)=>(
                  <li key={i}><a className="text-brand-700 underline" href={`/backend/files/${a}`} target="_blank" rel="noreferrer">{a}</a></li>
                ))}
              </ul>
            </div>
            {notes && (
              <div>
                <h3 className="font-semibold">Notes</h3>
                <p className="text-sm text-slate-600">{notes}</p>
              </div>
            )}
          </div>
        )}
      </FancyCard>
    </div>
  );
}
