import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score

from src.data import create_dataloader
from src.models import CXRModel
from src.utils import Config, run_inference
from src.utils.inference import find_optimal_thresholds


def main(model_path):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_info = torch.load(model_path)

    model = CXRModel(**save_info["config"])
    model.load_state_dict(save_info["model"])
    model.to(device)

    print('Model Successfully Loaded')

    model.eval()

    loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
    )

    print("Total X-rays to be tested:", len(loader.dataset))

    ablated_results = []

    preds, labels = run_inference(model=model, test_loader=loader)

    ablated_results.append({
        'Feature Index': '5',
        'Feature Name': 'All',
        'Predictions': preds,
        'Labels': labels
    })

    print("Baseline AUROCs added")

    tabular_features = ["Age", "Gender", "View Position", "Follow Up #"]

    ablated_feature_values = {
        0: 0.5,
        1: 0.5,
        2: 0.5,
        3: 0
    }

    actual_forward = model.forward

    for feat_idx, feat_name in enumerate(tabular_features):
        print("Ablating Feature:", feat_name)

        def ablated_forward_creator(actual_forward, idx):
            # Nesting is required to prevent recursion
            def ablated_forward_nested(img_batch, tabular_batch):
                ablated_tabular_batch = tabular_batch.clone()
                ablated_tabular_batch[:, idx] = ablated_feature_values[idx]
                return actual_forward(img_batch, ablated_tabular_batch)

            return ablated_forward_nested

        ablated_forward = ablated_forward_creator(actual_forward, feat_idx)

        model.forward = ablated_forward

        preds, labels = run_inference(model=model, test_loader=loader)

        model.forward = actual_forward  # Restore model.forward

        ablated_results.append({
            'Feature Index': feat_idx,
            'Feature Name': feat_name,
            'Predictions': preds,
            'Labels': labels
        })


    auroc_results=[]

    class_names = [name for name in cfg.class_labels]

    tabular_features = ["None", "Age", "Gender", "View Position", "Follow Up #"]

    for results in ablated_results:
        feature_name = results['Feature Name']
        print("Calculating Ablated AUROCs for", feature_name)
        if feature_name == "All":
            feature_name = "None"
        labels = results['Labels']
        preds = results['Predictions']

        thresholds = find_optimal_thresholds(labels, preds, cfg.class_labels)
        binary_preds = np.zeros_like(preds)
        for i, label in enumerate(cfg.class_labels):
            binary_preds[:, i] = (preds[:, i] >= thresholds[label]).astype(int)


        class_aurocs = []
        # We will calculate One vs All AUROC

        for class_idx in range(15):
            binary_labels = labels[:, class_idx]
            binary_preds_slice = binary_preds[:, class_idx]

            auroc = roc_auc_score(binary_labels, binary_preds_slice)
            class_name = class_names[class_idx]
            class_aurocs.append((class_name, auroc))

        auroc_results.append((feature_name, class_aurocs))


    auroc_df = pd.DataFrame(index=class_names, columns=tabular_features)
    for feature_name, class_aurocs in auroc_results:
        for class_name, auroc in class_aurocs:
            auroc_df.loc[class_name, feature_name] = auroc

    auroc_df = auroc_df.astype(float)
    auroc_df_transposed = auroc_df.T

    auroc_df_diff = auroc_df_transposed.subtract(auroc_df_transposed.loc['None'], axis=1)



    plt.figure(figsize=(15, 8))
    sns.heatmap(auroc_df_diff, annot=auroc_df_transposed, cmap="Spectral", fmt=".3f", cbar=False)
    plt.title('Ablation Study', fontsize=16)
    plt.ylabel('Ablated Feature', fontsize=14)
    plt.xlabel('Class', fontsize=18)

    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right') #https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks

    plt.tight_layout()

    plt.savefig('ablation_study_heatmap.png', dpi =300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specify path to saved model .pth file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/cs7643_final_share/emad_results/best_model_vit_b_32_mm_focal.pth",
        help="Specify path to saved model .pth file",
    )

    args = parser.parse_args()

    main(model_path=args.model_path)