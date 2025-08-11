// frontend/src/components/TrainPanel.tsx
import React, { useMemo, useState } from "react";
import { detectColumns, trainModel } from "../api";

const modelOptions = [
  { label: "Random Forest", value: "random_forest" },
  { label: "Logistic Regression (classification)", value: "logreg" },
  { label: "Neural Net (MLP)", value: "mlp" },
];

export default function TrainPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [model, setModel] = useState<string>("random_forest");
  const [saveDir, setSaveDir] = useState<string>("/app/outputs");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  async function onPickFile(f: File | null) {
    setFile(f);
    setColumns([]);
    setFeatures([]);
    setTarget("");
    setResult(null);
    if (!f) return;
    try {
      const cols = await detectColumns(f);
      setColumns(cols);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Failed to detect columns");
    }
  }

  function toggleFeature(col: string) {
    setFeatures(prev => (prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]));
  }

  async function onTrain() {
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    if (!features.length || !target) {
      setError("Select at least one feature and a target column.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await trainModel({
        file,
        features,
        target,
        model,
        saveDir,
      });
      setResult(data);
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        `Training failed (HTTP ${e?.response?.status ?? "?"})`;
      setError(msg);
      alert(`Training failed: ${msg}`);
    } finally {
      setBusy(false);
    }
  }

  const featureBoxes = useMemo(
    () =>
      columns.map(c => (
        <label key={c} style={{ display: "inline-flex", gap: 8, alignItems: "center", marginRight: 12 }}>
          <input
            type="checkbox"
            checked={features.includes(c)}
            onChange={() => toggleFeature(c)}
            disabled={c === target}
          />
          {c}
        </label>
      )),
    [columns, features, target]
  );

  return (
    <div className="p-4 space-y-4">
      <div className="rounded-xl p-4 bg-neutral-900/40 border border-neutral-800 space-y-3">
        <div className="text-lg font-semibold">Train Models</div>

        <div className="flex items-center gap-3">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={e => onPickFile(e.target.files?.[0] ?? null)}
          />
          <input
            className="px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700 w-72"
            placeholder="Save directory (in container)"
            value={saveDir}
            onChange={e => setSaveDir(e.target.value)}
          />
        </div>

        <div>
          <div className="text-sm opacity-80 mb-1">Features</div>
          <div className="flex flex-wrap">{featureBoxes}</div>
        </div>

        <div className="flex items-center gap-2">
          <div className="text-sm opacity-80">Target</div>
          <select
            className="px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700"
            value={target}
            onChange={e => setTarget(e.target.value)}
          >
            <option value="">— choose —</option>
            {columns.map(c => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>

          <div className="text-sm opacity-80 ml-4">Model</div>
          <select
            className="px-3 py-2 rounded-lg bg-neutral-800 border border-neutral-700"
            value={model}
            onChange={e => setModel(e.target.value)}
          >
            {modelOptions.map(m => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>

          <button
            onClick={onTrain}
            disabled={busy}
            className="ml-auto px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50"
          >
            {busy ? "Training…" : "Train"}
          </button>
        </div>

        {error && <div className="text-red-400 text-sm">{error}</div>}

        {result && (
          <div className="text-sm space-y-1">
            <div>Saved: <code>{result.model_path}</code></div>
            <div>Type: {result.model_type}</div>
            <div>Metrics: <code>{JSON.stringify(result.metrics)}</code></div>
            <div>
              Train/Test sizes: {result.n_train}/{result.n_test}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
