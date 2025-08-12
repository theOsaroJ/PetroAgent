import React, { useState } from 'react'
import { detectColumns, trainModel } from '../api'
import type { TrainResponse } from '../types'

export default function Trainer() {
  const [file, setFile] = useState<File | null>(null)
  const [columns, setColumns] = useState<string[]>([])
  const [features, setFeatures] = useState<string[]>([])
  const [target, setTarget] = useState<string>('')
  const [modelType, setModelType] = useState('random_forest')
  const [saveDir, setSaveDir] = useState('/app/outputs')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<TrainResponse | null>(null)

  async function handleDetect() {
    if (!file) return alert('Please choose a CSV first.')
    try {
      const cols = await detectColumns(file)
      setColumns(cols)
      setFeatures(cols.filter(c => c !== cols[0]))
      setTarget(cols[0] || 'y')
    } catch (e:any) {
      alert('Failed to detect columns: ' + (e.response?.data?.detail || e.message))
    }
  }

  async function handleTrain() {
    if (!file) return alert('Choose a CSV first.')
    if (!target) return alert('Pick a target column.')
    if (features.length === 0) return alert('Pick at least one feature.')

    setBusy(true)
    setResult(null)
    try {
      const res: TrainResponse = await trainModel({
        file, features, target,
        model_type: modelType,
        save_dir: saveDir
      })
      setResult(res)
      alert('Training complete. Model saved to: ' + res.saved_to)
    } catch (e:any) {
      const status = e?.response?.status
      const msg = e?.response?.data?.detail || e.message
      alert(`Training failed: ${status ? 'HTTP '+status+' - ' : ''}${msg}`)
      console.error(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div className="space-y-4">
        <div className="p-4 border border-slate-800 rounded-2xl">
          <div className="font-semibold mb-2">Train Models</div>
          <div className="flex items-center gap-3 mb-2">
            <input type="file" accept=".csv" onChange={e=>setFile(e.target.files?.[0]||null)} />
            <button onClick={handleDetect} className="px-3 py-1 rounded-lg bg-slate-800">Detect columns</button>
          </div>

          {columns.length>0 && (
            <div className="space-y-3">
              <div>
                <div className="text-sm mb-1">Features</div>
                <div className="flex flex-wrap gap-2 max-h-40 overflow-auto">
                  {columns.map(c => (
                    <label key={c} className="text-sm flex items-center gap-1 bg-slate-800 rounded px-2 py-1">
                      <input type="checkbox" checked={features.includes(c)} disabled={c===target}
                        onChange={(e)=>{
                          if (e.target.checked) setFeatures([...features, c])
                          else setFeatures(features.filter(f=>f!==c))
                        }} />
                      {c}
                    </label>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="text-sm">Target</div>
                <select className="bg-slate-800 rounded px-2 py-1" value={target} onChange={e=>setTarget(e.target.value)}>
                  {columns.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>

              <div className="flex items-center gap-3">
                <div className="text-sm">Model</div>
                <select className="bg-slate-800 rounded px-2 py-1" value={modelType} onChange={e=>setModelType(e.target.value)}>
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="gaussian_process">Gaussian Process</option>
                  <option value="mlp">Neural Net (MLP)</option>
                  <option value="transformer">Transformer (fallback to MLP if torch not installed)</option>
                </select>
              </div>

              <div className="flex items-center gap-3">
                <div className="text-sm">Save directory (in container)</div>
                <input className="bg-slate-800 rounded px-2 py-1" value={saveDir} onChange={e=>setSaveDir(e.target.value)} />
              </div>

              <button onClick={handleTrain} disabled={busy} className="px-3 py-1 rounded-lg bg-emerald-600 disabled:opacity-50">
                {busy?'Training...':'Train'}
              </button>
            </div>
          )}
        </div>

        {result && (
          <div className="p-4 border border-slate-800 rounded-2xl">
            <div className="font-semibold mb-2">Result</div>
            <pre className="text-sm">{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="p-4 border border-slate-800 rounded-2xl">
        <div className="font-semibold mb-2">Tips</div>
        <ul className="list-disc ml-6 text-sm space-y-1 text-slate-300">
          <li>Make sure your CSV has a single header row and numeric columns for features.</li>
          <li>For large files, the proxy allows up to 20MB by default (tweak in nginx.conf).</li>
          <li>Models are saved inside the <code>/app/outputs</code> folder of the ML container by default.</li>
        </ul>
      </div>
    </div>
  )
}
