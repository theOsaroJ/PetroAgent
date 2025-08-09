import React from "react";

export default function PlotGallery({ paths }: { paths: string[] }) {
  if (!paths.length) return (
    <div className="glass p-4 h-full flex items-center justify-center text-petro-muted">
      Plots and reports will appear here after training.
    </div>
  );
  return (
    <div className="glass p-4 h-full overflow-auto">
      <div className="text-petro-text font-semibold text-lg mb-3">Artifacts</div>
      <div className="grid md:grid-cols-2 gap-4">
        {paths.map((p,i)=>(
          <div key={i} className="bg-white/5 rounded-xl p-3">
            <div className="text-petro-muted text-xs truncate mb-2">{p}</div>
            <img src={p.replace("/app/data","/ml")} className="w-full rounded-lg border border-white/10"/>
          </div>
        ))}
      </div>
    </div>
  )
}
