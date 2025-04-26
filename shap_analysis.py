import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import run_inference
from src.data import create_dataloader
from src.utils.config import Config
from src.utils.persistence import load_model

# Load config and model
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "results/models/best_model_densenet121.pth"
model = load_model(model_path).to(device)
model.eval()

# Create test dataloader
loader = create_dataloader(
    clinical_data=cfg.tabular_clinical_test,
    cxr_images_dir=cfg.cxr_test_dir,
)

# Get one batch (images, tabular, labels)
images, tabular, labels = next(iter(loader))
images, tabular = images.to(device), tabular.to(device)

# Select background data for SHAP (first 100 tabular samples)
background = tabular[:100].to(device)

# Define a wrapper function if needed to isolate tabular forward
# Define a wrapper function for SHAP
def model_tabular_only(tabular_batch):
    if isinstance(tabular_batch, np.ndarray):
        tabular_batch = torch.tensor(tabular_batch, dtype=torch.float32)

    tabular_batch = tabular_batch.to(device)
    repeated_images = images[0].unsqueeze(0).repeat(tabular_batch.shape[0], 1, 1, 1)
    return model(repeated_images.to(device), tabular_batch).detach().cpu().numpy()

# SHAP KernelExplainer
explainer = shap.KernelExplainer(model_tabular_only, background.cpu().numpy())

# Explain one sample
shap_values = explainer.shap_values(tabular[0:1].cpu().numpy())

# Plot
#print(dir(cfg))
#print("type(cfg.tabular_clinical_test) =", type(cfg.tabular_clinical_test))

# Load the clinical test data from the file path
# Load the clinical test data from the file path
tabular_test_data = pd.read_csv(cfg.tabular_clinical_test)

# Extract only the first 4 columns (tabular data: imageIndex, followUpNumber, patientAge, patientGender)
#tabular_data = tabular_test_data.iloc[:, :4].values  # Get the first 4 columns
tabular_data = tabular_test_data.iloc[:, :].values

# Get the feature names for the first 4 columns (tabular data columns)
#tabular_feature_names = tabular_test_data.columns[:4].tolist()
tabular_feature_names = tabular_test_data.columns[:].tolist()

# Now calculate SHAP values (adjust the wrapper function if needed for tabular data)
# Aggregate SHAP values (average along axis 2)
shap_values_aggregated = np.mean(shap_values, axis=2)

# Print the new shape for debugging
print("Aggregated SHAP values shape:", shap_values_aggregated.shape)

# Now plot the summary plot
shap.summary_plot(shap_values_aggregated, tabular_data[0:1], feature_names=tabular_feature_names)

# Save the plot as a file
plt.savefig("shap_summary_plot.png")
