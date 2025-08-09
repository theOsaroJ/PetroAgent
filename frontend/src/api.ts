import axios from "axios";

export async function sendChat(message: string, history: {role: string, content: string}[]) {
  const { data } = await axios.post("/api/chat", { message, history });
  return data.reply as string;
}

export async function uploadCSV(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axios.post("/api/upload", form, { headers: { "Content-Type": "multipart/form-data" } });
  return data as { file_id: string, path: string };
}

export async function listModels() {
  const { data } = await axios.get("/api/models");
  return data.models as string[];
}

export async function trainModel(args: {
  file_id?: string, path?: string,
  features: string[], target: string,
  model: string, save_path?: string
}) {
  const { data } = await axios.post("/api/train", args);
  return data as { model: string, save_path: string, scaler_included: boolean, metrics: any };
}

export async function generatePlots(args: { file_id?: string, path?: string, features: string[], target: string }) {
  const { data } = await axios.post("/api/plots", args);
  return data.images as Record<string, string>;
}

export async function describeData(args: { file_id?: string, path?: string }) {
  const { data } = await axios.post("/api/describe", args);
  return data as { columns: string[], preview: any[], describe: any };
}
