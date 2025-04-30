import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def main(csv_path):
    df = pd.read_csv(csv_path)

    feature_cols = ["patientAge", "patientGender", "viewPosition", "followUpNumber"]

    # class_cols = [col for col in df.columns if col.startswith('label_')]

    class_cols = [
        "label_atelectasis",
        "label_cardiomegaly",
        "label_consolidation",
        "label_edema",
        "label_effusion",
        "label_emphysema",
        "label_fibrosis",
        "label_hernia",
        "label_infiltration",
        "label_mass",
        "label_no_finding",
        "label_nodule",
        "label_pleural_thickening",
        "label_pneumonia",
        "label_pneumothorax",
    ]

    feature_names = [
        "Age",
        "Gender",
        "View Position",
        "Follow Up #",
    ]

    class_names = [col.replace("label_", "") for col in class_cols]

    corr_matrix = pd.DataFrame(
        0.0, index=feature_names, columns=class_names, dtype=float
    )

    for i, feature in enumerate(feature_cols):
        for j, class_col in enumerate(class_cols):
            corr, _ = pearsonr(df[feature], df[class_col])
            corr_matrix.loc[feature_names[i], class_names[j]] = corr

    plt.figure(figsize=(15, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="Spectral", fmt=".3f", cbar=False)
    plt.title("Pearson Correlation Matrix", fontsize=16)
    plt.ylabel("Class", fontsize=14)
    plt.xlabel("Tabular Feature", fontsize=18)

    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig("pearson_matrix.png", dpi=300)

    empty_row = pd.Series(0.0, index=class_names)
    corr_matrix = pd.concat([pd.DataFrame([empty_row], index=["None"]), corr_matrix])

    plt.figure(figsize=(15, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="Spectral", fmt=".3f", cbar=False)
    plt.title("Pearson Correlation Matrix", fontsize=16)
    plt.ylabel("Class", fontsize=14)
    plt.xlabel("Tabular Feature", fontsize=18)

    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig("pearson_matrix_with_None.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify path to saved .csv file")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="artifacts/test.csv",
        help="Specify path to .csv file",
    )

    args = parser.parse_args()

    main(csv_path=args.csv_path)
