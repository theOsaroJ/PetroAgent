import React from "react";

export default function PlotGallery({ images }: { images: Record<string,string> }) {
  const keys = Object.keys(images);
  if (keys.length === 0) return (
    <div className="bg-white rounded-2xl shadow-lg p-4 text-slate-500">
      Plots will appear here after you click "Generate Plots".
    </div>
  );

  return (
    <div className="bg-white rounded-2xl shadow-lg p-4">
      <h2 className="text-xl font-semibold mb-3">Analysis & Plots</h2>
      <div className="grid md:grid-cols-2 gap-4">
        {keys.map(k => (
          <div key={k} className="border rounded-lg p-2">
            <div className="text-xs text-slate-600 mb-1">{k}</div>
            <img src={`data:image/png;base64,${images[k]}`} className="w-full rounded-md" />
          </div>
        ))}
      </div>
    </div>
  );
}
