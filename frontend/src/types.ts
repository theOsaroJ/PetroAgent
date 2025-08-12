export type TrainResponse = {
  model_type: string;
  saved_to: string;
  n_samples: number;
  metrics: { rmse: number; r2: number; }
}
