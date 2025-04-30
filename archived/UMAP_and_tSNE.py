import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys

project_root = "/home/wasabi/PycharmProjects/cs-7643-final-project"
sys.path.insert(0, project_root)

from src.data import create_dataloader
from src.models import CXRModel
from src.utils import Config

def main(vanilla_path, multimodal_path, selected_classes):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
    )



    vanilla_save_info = torch.load(vanilla_path, map_location=device)
    vanilla_model = CXRModel(**vanilla_save_info["config"])
    vanilla_model.load_state_dict(vanilla_save_info["model"])
    vanilla_model.to(device)
    vanilla_model.eval()

    multimodal_save_info = torch.load(multimodal_path, map_location=device)
    multimodal_model = CXRModel(**multimodal_save_info["config"])
    multimodal_model.load_state_dict(multimodal_save_info["model"])
    multimodal_model.to(device)
    multimodal_model.eval()

    def get_preclassifier_hook(alist):
        def hook(model, input):
            alist.append(input[0].detach().cpu())
        return hook

    vanilla_features = []
    multimodal_features = []
    all_labels = []

    hook_handle_vanilla = vanilla_model.model.classifier.register_forward_pre_hook(
        get_preclassifier_hook(vanilla_features)
    )

    hook_handle_multimodal = multimodal_model.model.classifier.register_forward_pre_hook(
        get_preclassifier_hook(multimodal_features)
    )

    with torch.no_grad():
        for images, tabular, labels in tqdm(loader):
            images = images.to(device)
            tabular = tabular.to(device)
            all_labels.append(labels.numpy())

            _ = vanilla_model(images, tabular)
            _ = multimodal_model(images, tabular)

    hook_handle_vanilla.remove()
    hook_handle_multimodal.remove()

    vanilla_features_tensor = torch.cat(vanilla_features, dim=0)
    multimodal_features_tensor = torch.cat(multimodal_features, dim=0)
    vanilla_features = vanilla_features_tensor.numpy()
    multimodal_features = multimodal_features_tensor.numpy()
    all_labels = np.vstack(all_labels)
    dominant_labels = np.argmax(all_labels, axis=1)

    class_mask = np.isin(dominant_labels, selected_classes)
    filtered_vanilla_features = vanilla_features[class_mask]
    filtered_multimodal_features = multimodal_features[class_mask]
    filtered_labels = dominant_labels[class_mask]

    class_names = [name for name in cfg.class_labels]
    selected_class_names = [class_names[idx] for idx in selected_classes]

    class_map = {cls: i for i, cls in enumerate(selected_classes)}
    mapped_labels = np.array([class_map[label] for label in filtered_labels])

    vanilla_reducer = umap.UMAP(
        n_neighbors=75,
        min_dist = 0.1,
        random_state=69
    )
    vanilla_embedding = vanilla_reducer.fit_transform(filtered_vanilla_features)

    multimodal_reducer = umap.UMAP(
        n_neighbors=75,
        min_dist = 0.1,
        random_state=69
    )
    multimodal_embedding = multimodal_reducer.fit_transform(filtered_multimodal_features)

    vanilla_tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=69,
        n_iter=1000
    ).fit_transform(filtered_vanilla_features)

    multimodal_tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=69,
        n_iter=1000
    ).fit_transform(filtered_multimodal_features)


    selected_class_names = [class_names[idx] for idx in selected_classes]

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab20', len(selected_class_names))
    for i, class_name in enumerate(selected_class_names):
        mask = mapped_labels == i
        plt.scatter(
            vanilla_embedding[mask, 0],
            vanilla_embedding[mask, 1],
            color=cmap(i),
            s=5,
            alpha=0.7,
            label=class_name
        )
    plt.title('UMAP Projection Without Tabular Data')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Classes')
    plt.tight_layout()
    plt.savefig('umap_vanilla.png', dpi=300)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(selected_class_names):
        mask = mapped_labels == i
        plt.scatter(
            multimodal_embedding[mask, 0],
            multimodal_embedding[mask, 1],
            color=cmap(i),
            s=5,
            alpha=0.7,
            label=class_name
        )
    plt.title('UMAP Projection With Tabular Data')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Classes')
    plt.tight_layout()
    plt.savefig('umap_multimodal.png', dpi=300)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(selected_class_names):
        mask = mapped_labels == i
        plt.scatter(
            vanilla_tsne[mask, 0],
            vanilla_tsne[mask, 1],
            color=cmap(i),
            s=5,
            alpha=0.7,
            label=class_name
        )
    plt.title('t-SNE Projection Without Tabular Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Classes')
    plt.tight_layout()
    plt.savefig('tsne_vanilla.png', dpi=300)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(selected_class_names):
        mask = mapped_labels == i
        plt.scatter(
            multimodal_tsne[mask, 0],
            multimodal_tsne[mask, 1],
            color=cmap(i),
            s=5,
            alpha=0.7,
            label=class_name
        )
    plt.title('t-SNE Projection With Tabular Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Classes')
    plt.tight_layout()
    plt.savefig('tsne_multimodal.png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specify path to saved model .pth files"
    )
    parser.add_argument(
        "--multimodal_path",
        type=str,
        default="/tmp/cs7643_final_share/emad_results/best_model_vit_b_32_mm_focal.pth",
        help="Specify path to saved multimodal model .pth file",
    )
    parser.add_argument(
        "--vanilla_path",
        type=str,
        default="/tmp/cs7643_final_share/emad_results/best_model_vit_b_32_focal.pth",
        help="Specify path to saved image only model .pth file",
    )
    parser.add_argument(
        "--selected_classes",
        type=int,
        nargs='+',
        default=[0,4,8,11],
        help="Specify the class indices selected for UMAP and tSNE",
    )

    args = parser.parse_args()

    main(vanilla_path=args.vanilla_path, multimodal_path=args.multimodal_path, selected_classes=args.selected_classes)