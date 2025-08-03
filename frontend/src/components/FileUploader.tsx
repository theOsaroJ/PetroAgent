import React from "react";
import Papa from "papaparse";

interface Props {
  onDetect: (info: any) => void;
}

export default function FileUploader({ onDetect }: Props) {
  const fileInput = React.useRef<HTMLInputElement>(null);
  const [headers, setHeaders] = React.useState<string[]>([]);
  const [config, setConfig] = React.useState<{ inputs: string[]; target: string }>({
    inputs: [],
    target: ""
  });

  const handleFile = () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const fields = results.meta.fields || [];
        setHeaders(fields);
        onDetect({ filename: file.name, headers: fields, preview: results.data.slice(0,5) });
      }
    });
  };

  const confirmSchema = () => {
    onDetect({ ...config });
  };

  return (
    <div className="mb-4">
      <input type="file" ref={fileInput} onChange={handleFile} className="mb-2" />
      {headers.length > 0 && (
        <div>
          <label className="block mb-1">Features (ctrl+click multi):</label>
          <select
            multiple
            className="border w-full p-1 mb-2"
            onChange={(e) =>
              setConfig((c) => ({
                ...c,
                inputs: Array.from(e.target.selectedOptions).map((o) => o.value)
              }))
            }
          >
            {headers.map((h) => (
              <option key={h}>{h}</option>
            ))}
          </select>
          <label className="block mb-1">Target:</label>
          <select
            className="border w-full p-1 mb-2"
            onChange={(e) =>
              setConfig((c) => ({ ...c, target: e.target.value }))
            }
            value={config.target}
          >
            <option value="">-- choose --</option>
            {headers.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded"
            onClick={confirmSchema}
          >
            Confirm Schema
          </button>
        </div>
      )}
    </div>
  );
}
