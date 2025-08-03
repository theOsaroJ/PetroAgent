import React from "react";

export default function ModelSelector({ modelType, setModelType }) {
  return (
    <div className="mb-4">
      <div className="font-semibold">Select model:</div>
      <select value={modelType} onChange={e => setModelType(e.target.value)}>
        <option value="neural">Neural Network</option>
        <option value="random_forest">Random Forest</option>
        <option value="gp">Gaussian Process</option>
        <option value="transformer">Transformer</option>
      </select>
    </div>
  );
}
