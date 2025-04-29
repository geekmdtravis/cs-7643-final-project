import os

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from tqdm import tqdm

from src.data import create_dataloader
from src.utils.config import Config
from src.utils.persistence import load_model

MAX_SAMPLES = 10
TABULAR_BACKGROUND_SIZE = 100
IMAGE_BACKGROUND_SIZE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()

loader = create_dataloader(
    clinical_data=cfg.tabular_clinical_test,
    cxr_images_dir=cfg.cxr_test_dir,
)

all_images, all_tabular, all_labels = [], [], []
for batch in loader:
    imgs, tabs, lbls = batch
    all_images.append(imgs)
    all_tabular.append(tabs)
    all_labels.append(lbls)
    if MAX_SAMPLES and len(torch.cat(all_images)) >= MAX_SAMPLES:
        break

images = torch.cat(all_images)[:MAX_SAMPLES].to(device)
tabular = torch.cat(all_tabular)[:MAX_SAMPLES].to(device)
labels = torch.cat(all_labels)[:MAX_SAMPLES].to(device)


def compute_shap_importance(model_path, model_name):
    model = load_model(model_path).to(device)
    model.eval()

    background_tabular = tabular[:TABULAR_BACKGROUND_SIZE]

    def model_tabular_only(tabular_batch):
        if isinstance(tabular_batch, np.ndarray):
            tabular_batch = torch.tensor(tabular_batch, dtype=torch.float32)
        tabular_batch = tabular_batch.to(device)
        repeated_image = images[0].unsqueeze(0).repeat(tabular_batch.size(0), 1, 1, 1)
        return model(repeated_image, tabular_batch).detach().cpu().numpy()

    explainer_tab = shap.KernelExplainer(
        model_tabular_only, background_tabular.cpu().numpy()
    )

    shap_values_tab_list = []
    for i in tqdm(range(len(tabular)), desc=f"{model_name} Tabular SHAP"):
        sample = tabular[i].unsqueeze(0).cpu().numpy()
        sv = explainer_tab.shap_values(sample)
        shap_values_tab_list.append(sv)

    shap_tab_np = np.array(shap_values_tab_list)
    shap_tab_mean = np.mean(np.abs(shap_tab_np), axis=(0, 2))
    tabular_importance = np.mean(shap_tab_mean)

    background_images = images[:IMAGE_BACKGROUND_SIZE]
    fixed_tabular = tabular[0]

    class ImageOnlyModelWrapper(torch.nn.Module):
        def __init__(self, model, fixed_tabular):
            super().__init__()
            self.model = model
            self.fixed_tabular = fixed_tabular.to(device)

        def forward(self, x_img):
            repeated_tab = self.fixed_tabular.unsqueeze(0).repeat(x_img.size(0), 1)
            return self.model(x_img, repeated_tab)

    wrapped_model = ImageOnlyModelWrapper(model, fixed_tabular)
    explainer_img = shap.GradientExplainer(wrapped_model, background_images)

    shap_values_img_list = []
    for i in tqdm(range(len(images)), desc=f"{model_name} Image SHAP"):
        img_sample = images[i].unsqueeze(0)
        shap_vals = explainer_img.shap_values(img_sample)
        shap_values_img_list.append([np.abs(sv).mean() for sv in shap_vals])

    shap_img_np = np.array(shap_values_img_list)
    shap_img_mean = np.mean(shap_img_np)

    return tabular_importance, shap_img_mean


dn_tab, dn_img = compute_shap_importance(
    "/tmp/cs7643_final_share/best_models/densenet121_mm_best.pth", "DenseNet121"
)
vit_tab, vit_img = compute_shap_importance(
    "/tmp/cs7643_final_share/best_models/vit_b_32_mm_best.pth", "ViT-B/32"
)

labels = ["Tabular", "Image"]
dn_values = [dn_tab, dn_img]
vit_values = [vit_tab, vit_img]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width / 2, dn_values, width, label="DenseNet121")
bars2 = ax.bar(x + width / 2, vit_values, width, label="ViT-B/32")

ax.set_yscale("log")
ax.set_ylabel("Mean SHAP Magnitude (log scale)", fontsize=12, fontweight="bold")
ax.set_xlabel("Modality", fontsize=14, fontweight="bold")
ax.set_title(
    "SHAP Importance: Tabular vs Image (Multimodal Models)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(1e-4, 1e-1)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2e}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

plt.tight_layout()
os.makedirs("results/experiment9", exist_ok=True)
plt.savefig(
    "results/experiment9/shap_multimodal_comparison_densenet_vs_vit.pdf",
    dpi=500,
    format="pdf",
)
