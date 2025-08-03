const axios = require('axios');
const ML_SERVICE_URL = 'http://ml_service:8000';

async function trainModel(type, filePath, targetColumn, featureColumns) {
  const resp = await axios.post(
    `${ML_SERVICE_URL}/train/${type}`,
    { file_path: filePath, target_column: targetColumn, feature_columns: featureColumns }
  );
  return resp.data;
}

async function predict(type, inputData) {
  const resp = await axios.post(
    `${ML_SERVICE_URL}/predict/${type}`,
    { data: inputData }
  );
  return resp.data;
}

module.exports = { trainModel, predict };
