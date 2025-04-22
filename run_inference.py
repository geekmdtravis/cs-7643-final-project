from src.data import create_dataloader
from src.utils import Config, evaluate_model, print_evaluation_results, run_inference

cfg = Config()

if __name__ == "__main__":
    print("Beginning inference...")
    path = "results/models/best_model_densenet121.pth"
    loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
    )
    preds, labels = run_inference(
        model=path,
        test_loader=loader,
    )

    auc_scores, report = evaluate_model(preds, labels)

    print_evaluation_results(
        auc_scores=auc_scores, report=report, save_path="results/evaluation_report.txt"
    )
    print("Inference completed.")
