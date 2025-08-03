import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState<File|null>(null);
  const [uploadPath, setUploadPath] = useState('');
  const [modelType, setModelType] = useState<'xgboost'|'gpr'|'classification'|'neural'>('xgboost');
  const [targetCol, setTargetCol] = useState('');
  const [featureCols, setFeatureCols] = useState('');
  const [msgs, setMsgs] = useState<{ user?:string; bot?:string }[]>([]);
  const [inp, setInp] = useState('');

  useEffect(() => {
    // no-op
  }, []);

  const uploadFile = async () => {
    if (!file) return;
    const form = new FormData(); form.append('file', file);
    const { data } = await axios.post('/api/upload', form, {
      headers: {'Content-Type':'multipart/form-data'}
    });
    setUploadPath(data.path);
  };

  const trainModel = async () => {
    if (!uploadPath||!targetCol||!featureCols)
      return alert('Upload CSV & specify columns');
    const cols = featureCols.split(',').map(c=>c.trim());
    const { data } = await axios.post(`/api/train/${modelType}`, {
      filePath: uploadPath,
      targetColumn: targetCol,
      featureColumns: cols
    });
    alert(`✅ Trained ${modelType}: score=${data.train_score}`);
  };

  const predictModel = async () => {
    const raw = prompt('Enter JSON array of feature objects:');
    if (!raw) return;
    const arr = JSON.parse(raw);
    const { data } = await axios.post(`/api/predict/${modelType}`, { inputData: arr });
    alert(`📈 Predictions:\n${JSON.stringify(data.predictions, null, 2)}`);
  };

  const sendChat = async () => {
    if (!inp.trim()) return;
    setMsgs(m=>[...m,{ user: inp }]);
    const msg = inp; setInp('');
    try {
      const resp = await axios.post('http://localhost:7000/agent/respond', { message: msg });
      setMsgs(m=>[...m,{ bot: resp.data.response }]);
    } catch (e:any) {
      setMsgs(m=>[...m,{ bot: `Error: ${e.message}` }]);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Hero */}
      <div
        className="h-48 bg-cover bg-center flex items-center"
        style={{ backgroundImage: "url(/images/banner.png)" }}
      >
        <img src="/images/logo.png" alt="Logo" className="h-16 ml-8" />
        <h1 className="text-white text-4xl font-bold ml-4 drop-shadow-lg">
          Petroleum Data Agent
        </h1>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Controls */}
        <aside className="w-1/3 bg-white p-6 overflow-y-auto space-y-8">
          <section>
            <h2 className="text-2xl font-semibold mb-2">📂 Upload CSV</h2>
            <input
              type="file" accept=".csv"
              onChange={e=>e.target.files&&setFile(e.target.files[0])}
              className="w-full mb-2"
            />
            <button
              onClick={uploadFile}
              className="w-full py-2 bg-green-600 text-white rounded shadow"
            >
              Upload
            </button>
            {uploadPath && (
              <p className="mt-2 text-sm text-gray-600 break-all">
                Path: {uploadPath}
              </p>
            )}
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-2">⚙️ Train & Predict</h2>
            <label className="block mb-1">Model:</label>
            <select
              value={modelType}
              onChange={e=>setModelType(e.target.value as any)}
              className="w-full mb-3 border rounded p-2"
            >
              <option value="xgboost">XGBoost Regressor</option>
              <option value="gpr">Gaussian Process</option>
              <option value="classification">Random Forest Classifier</option>
              <option value="neural">Neural Net</option>
            </select>
            <label className="block mb-1">Target Column:</label>
            <input
              type="text" value={targetCol}
              onChange={e=>setTargetCol(e.target.value)}
              className="w-full mb-3 border rounded p-2"
            />
            <label className="block mb-1">Features (comma-sep):</label>
            <input
              type="text" value={featureCols}
              onChange={e=>setFeatureCols(e.target.value)}
              className="w-full mb-3 border rounded p-2"
            />
            <div className="flex space-x-2">
              <button
                onClick={trainModel}
                className="flex-1 py-2 bg-blue-600 text-white rounded shadow"
              >
                Train
              </button>
              <button
                onClick={predictModel}
                className="flex-1 py-2 bg-yellow-500 text-white rounded shadow"
              >
                Predict
              </button>
            </div>
          </section>
        </aside>

        {/* Chat */}
        <main className="flex-1 flex flex-col bg-gray-100 p-6">
          <h2 className="text-2xl font-semibold mb-2">💬 Chat & Analyze</h2>
          <div className="flex-1 overflow-y-auto mb-4 p-4 bg-white rounded shadow">
            {msgs.map((m,i) => (
              <div key={i} className={m.user ? 'text-right' : 'text-left'}>
                <span className={`inline-block p-3 my-1 rounded ${
                  m.user ? 'bg-blue-200' : 'bg-gray-200'
                }`}>
                  {m.user ?? m.bot}
                </span>
              </div>
            ))}
          </div>
          <div className="flex">
            <input
              value={inp} onChange={e=>setInp(e.target.value)}
              className="flex-1 p-3 rounded-l border"
              placeholder="Ask the agent anything…"
            />
            <button
              onClick={sendChat}
              className="px-6 bg-indigo-600 text-white rounded-r shadow"
            >
              Send
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}
