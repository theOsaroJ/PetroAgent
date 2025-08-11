// frontend/src/api.ts
import axios from "axios";

const ml = axios.create({
  baseURL: "/api/ml",
  // IMPORTANT: don't set Content-Type for FormData; the browser sets the boundary.
});

export async function detectColumns(file: File): Promise<string[]> {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await ml.post("/columns", fd);
  return data.columns as string[];
}

export type TrainPayload = {
  file: File;
  features: string[];   // array of column names
  target: string;       // target column
  model?: string;       // "random_forest" | "logreg" | "mlp" | "rf"
  saveDir?: string;     // inside container, e.g. "/app/outputs"
};

export async function trainModel(p: TrainPayload) {
  const fd = new FormData();
  fd.append("file", p.file);
  fd.append("features", p.features.join(","));
  fd.append("target", p.target);
  fd.append("model", p.model ?? "random_forest");
  fd.append("save_dir", p.saveDir ?? "/app/outputs");

  const { data } = await ml.post("/train", fd);
  return data;
}
