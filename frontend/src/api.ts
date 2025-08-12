import axios from 'axios';

export async function chat(message: string): Promise<string> {
  const res = await axios.post('/api/chat', { message });
  return res.data.reply;
}

export async function detectColumns(file: File): Promise<string[]> {
  const form = new FormData();
  form.append('file', file);
  const res = await axios.post('/ml/columns', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data.columns;
}

export async function trainModel(params: {
  file: File;
  features: string[];
  target: string;
  model_type: string;
  save_dir: string;
}) {
  const form = new FormData();
  form.append('file', params.file);
  form.append('features', JSON.stringify(params.features));
  form.append('target', params.target);
  form.append('model_type', params.model_type);
  form.append('save_dir', params.save_dir);
  const res = await axios.post('/ml/train', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
}
