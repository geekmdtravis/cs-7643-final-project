# ChestX-ray14 Multi-Modal Deep Learning

This project explores the integration of tabular patient data with deep learning models for chest X-ray classification using the ChestX-ray14 dataset. We compare CNN and Vision Transformer architectures with two different methods of tabular data integration.

Will update this README with more details about the project.

## Example `.env` file (Root Directory)

```bash
NUM_WORKERS=32 # Number of workers for data loading
BATCH_SIZE=32 # Batch size for training
NUM_EPOCHS=10 # Number of epochs for training
LEARNING_RATE=0.001 # Learning rate for optimizer
OPTIMIZER=adam # Optimizer to use (adam, sgd)
DEVICE=cuda # cuda or cpu
LOG_LEVEL=debug # Logging level (debug, info, warning, error, criticalerror)
LOG_FILE=app.log # Log file name
```
