import axios from "axios";

export const chat = async (message: string) => {
  const r = await axios.post("/api/chat", { message });
  return r.data.reply as string;
};

export const uploadCSV = async (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  const r = await axios.post("/ml/upload", fd, { headers: { "Content-Type": "multipart/form-data" } });
  return r.data as { upload_id: string; n_rows: number; n_cols: number; columns: string[]; dtypes: Record<string, string> };
};

export type TrainPayload = {
  upload_id: string;
  target: string;
  features: string[];
  task: "regression"|"classification";
  model_type: "rf"|"gp"|"xgb"|"nn"|"transformer";
  output_dir: string;
  test_size?: number;
  val_size?: number;
  random_state?: number;
  standardize?: boolean;
  max_rows?: number | null;
};
export const trainModel = async (payload: TrainPayload) => {
  const r = await axios.post("/ml/train", payload);
  return r.data as { model_path: string; metrics: Record<string, number>; artifacts: string[] };
};
