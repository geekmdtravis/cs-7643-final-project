from src.data import create_dataloader
from src.utils import Config, evaluate_model, print_evaluation_results, run_inference

cfg = Config()

if __name__ == "__main__":
    print("Beginning inference...")
    loader = create_dataloader(
        clinical_data=cfg.tabular_clinical_test,
        cxr_images_dir=cfg.cxr_test_dir,
    )

    path = "results/models/best_model_densenet121.pth"
    preds, labels = run_inference(
        model=path,
        test_loader=loader,
    )

    results = evaluate_model(
        preds=preds,
        labels=labels,
    )
    # Print and save results
    print_evaluation_results(
        results=results,
        save_path="results/evaluation_report.txt",
    )
    print("Inference completed.")
