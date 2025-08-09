import axios from "axios";

export const api = axios.create({
  // same origin; nginx routes:
  // /api/* -> agent, /backend/* -> backend, /ml/* -> ml_service
  timeout: 300000 // long for training
});

// Chat
export async function chatLLM(message, contextSummary = "") {
  const res = await api.post("/api/chat", { message, context: contextSummary });
  return res.data;
}

// Upload CSV to backend
export async function uploadCSV(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/backend/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // {file_rel_path, columns, head}
}

// Train model
export async function trainModel(payload) {
  const res = await api.post("/ml/train", payload);
  return res.data; // {metrics, artifacts, model_path, notes}
}
