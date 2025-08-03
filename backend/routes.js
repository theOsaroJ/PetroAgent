const express = require('express');
const multer = require('multer');
const mlService = require('./services/mlService');
const upload = multer({ dest: 'uploads/' });
const router = express.Router();

// Upload CSV
router.post('/upload', upload.single('file'), (req, res) => {
  res.json({ path: req.file.path });
});

// Train model
router.post('/train/:modelType', async (req, res) => {
  try {
    const out = await mlService.trainModel(
      req.params.modelType,
      req.body.filePath,
      req.body.targetColumn,
      req.body.featureColumns
    );
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Predict
router.post('/predict/:modelType', async (req, res) => {
  try {
    const out = await mlService.predict(
      req.params.modelType,
      req.body.inputData
    );
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
