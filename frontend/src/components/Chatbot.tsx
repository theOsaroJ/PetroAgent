import React, { useState } from 'react';
import Papa from 'papaparse';
import axios from 'axios';

export default function FileUploader() {
  const [headers, setHeaders]   = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget]     = useState<string>('');
  const [algorithm, setAlg]     = useState<string>('neural');
  const [file, setFile]         = useState<File|null>(null);

  const onFile = (f: File) => {
    setFile(f);
    Papa.parse(f, {
      preview: 1,
      complete: (res) => setHeaders(res.data[0] as string[])
    });
  };

  const train = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    fd.append('algorithm', algorithm);
    fd.append('epochs', '20');
    const resp = await axios.post('http://localhost:8000/train', fd, {
      headers: {'Content-Type':'multipart/form-data'}
    });
    alert('Trained with ' + resp.data.algorithm);
  };

  return (
    <div>
      <input type="file" accept=".csv" onChange={e => e.target.files && onFile(e.target.files[0])} />
      {headers.length > 0 && (
        <>
          <h3>Select features</h3>
          <select multiple value={features} onChange={e => setFeatures(Array.from(e.target.selectedOptions).map(o=>o.value))}>
            {headers.map(h=> <option key={h}>{h}</option>)}
          </select>
          <h3>Select target</h3>
          <select value={target} onChange={e=>setTarget(e.target.value)}>
            <option value="">--</option>
            {headers.map(h=> <option key={h}>{h}</option>)}
          </select>
          <h3>Select algorithm</h3>
          <select value={algorithm} onChange={e=>setAlg(e.target.value)}>
            <option value="neural">Neural Network</option>
            <option value="rf">Random Forest</option>
            <option value="linear">Linear Regression</option>
            <option value="xgboost">XGBoost</option>
          </select>
          <button onClick={train} className="mt-2 p-2 bg-green-500 text-white rounded">
            Train {algorithm}
          </button>
        </>
      )}
    </div>
  );
}
