import React, { useState } from "react";
import axios from "axios";

export default function Upload({ onSchema, onSelect }) {
  const [file, setFile] = useState();
  const [cols, setCols] = useState([]);

  const handleUpload = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const { data } = await axios.post("/api/upload", fd);
    setCols(data.columns);
    onSchema(data.columns);
  };

  return (
    <div>
      <label className="block mb-2 font-semibold">Choose CSV:</label>
      <input
        type="file"
        accept=".csv"
        onChange={e => setFile(e.target.files[0])}
        className="mb-2"
      />
      <button
        onClick={handleUpload}
        className="px-4 py-1 bg-green-600 text-white rounded"
      >
        Upload
      </button>

      {cols.length > 0 && (
        <div className="mt-4">
          <label className="font-semibold">Features:</label>
          <select
            multiple
            className="w-full p-1 border rounded h-20"
            onChange={e =>
              onSelect(sel => ({
                ...sel,
                features: Array.from(e.target.selectedOptions, o => o.value)
              }))
            }
          >
            {cols.map(c => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.dtype})
              </option>
            ))}
          </select>

          <label className="font-semibold mt-2 block">Target:</label>
          <select
            className="w-full p-1 border rounded"
            onChange={e =>
              onSelect(sel => ({ ...sel, target: e.target.value }))
            }
          >
            <option value="">-- select --</option>
            {cols.map(c => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
