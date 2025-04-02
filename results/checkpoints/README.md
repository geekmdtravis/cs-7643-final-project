# Model Checkpoints

This directory contains saved model checkpoints from training runs.

## Structure

Checkpoints are organized by model type and experiment:

```
checkpoints/
├── cnn_baseline/
├── vit_baseline/
├── cnn_embed/
├── vit_embed/
├── cnn_fusion/
└── vit_fusion/
```

Each checkpoint includes model weights and training state for reproducibility.

Note: Due to file size, model checkpoints are not tracked in git. This directory structure is maintained for organization.
