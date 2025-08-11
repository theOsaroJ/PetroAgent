import React, { useState } from "react";

type Column = string;

export default function Trainer() {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<Column[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [model, setModel] = useState<string>("Random Forest");
  const [saveDir, setSaveDir] = useState<string>("/app/outputs");
  const [working, setWorking] = useState<boolean>(false);

  async function onDetect() {
    if (!file) {
      alert("Choose a CSV first");
      return;
    }
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/ml/columns", {
      method: "POST",
      body: fd, // DO NOT set Content-Type
    });
    if (!r.ok) {
      const t = await r.text();
      alert(`Detect failed: ${t}`);
      return;
    }
    const data = await r.json();
    setColumns(data.columns || []);
  }

  async function onTrain() {
    if (!file) {
      alert("Choose a CSV first");
      return;
    }
    if (!target) {
      alert("Pick a target column");
      return;
    }
    if (selected.length === 0) {
      alert("Pick at least one feature");
      return;
    }

    setWorking(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("features", selected.join(","));
      fd.append("target", target);
      fd.append("model_name", model);
      fd.append("save_dir", saveDir);

      const resp = await fetch("/ml/train", {
        method: "POST",
        body: fd, // DO NOT set Content-Type
      });

      if (!resp.ok) {
        const txt = await resp.text();
        alert(`Training failed: ${resp.status} ${txt}`);
        return;
      }
      const data = await resp.json();
      alert(`OK! Saved: ${data.model_path}\nR^2=${data.metrics.r2.toFixed(3)} RMSE=${data.metrics.rmse.toFixed(3)}`);
    } catch (e: any) {
      alert(`Training error: ${e?.message || e}`);
    } finally {
      setWorking(false);
    }
  }

  return (
    <div className="trainer">
      <div>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button onClick={onDetect}>Detect columns</button>
      </div>

      {columns.length > 0 && (
        <>
          <div style={{ marginTop: 12 }}>
            <strong>Features</strong>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {columns.map((c) => (
                <label key={c} style={{ display: "inline-flex", gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={selected.includes(c)}
                    onChange={(e) =>
                      setSelected((prev) =>
                        e.target.checked
                          ? [...prev, c]
                          : prev.filter((x) => x !== c)
                      )
                    }
                  />
                  {c}
                </label>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <strong>Target</strong>
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">-- choose --</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginTop: 12 }}>
            <strong>Model</strong>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option>Random Forest</option>
              <option>Linear Regression</option>
            </select>
          </div>

          <div style={{ marginTop: 12 }}>
            <strong>Save directory (in container)</strong>
            <input
              value={saveDir}
              onChange={(e) => setSaveDir(e.target.value)}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ marginTop: 16 }}>
            <button onClick={onTrain} disabled={working}>
              {working ? "Training..." : "Train"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
