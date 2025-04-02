# Visualization Plots

This directory contains generated plots and visualizations from experiments.

## Types of Plots

- Training/validation curves
- Confusion matrices
- ROC curves
- Attention visualization maps (for ViT models)
- Performance comparison plots
- Feature importance visualizations

## Organization

Plots are organized by experiment and visualization type:

```
plots/
├── training/          # Learning curves, loss plots
├── evaluation/        # Confusion matrices, ROC curves
├── attention_maps/    # ViT attention visualizations
└── comparisons/       # Cross-model performance comparisons
```

Note: Plot files are tracked in git for documentation purposes, but please be mindful of file sizes when committing new visualizations.
