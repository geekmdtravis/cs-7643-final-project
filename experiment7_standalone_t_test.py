"""
This file assumes that the target files are already present,
having been tuned and files saved. There are many implicit assumptions
here as the implementation is tightly coupled to
'experiment7.py', which itself is based on 'run_tuner.py'. So,
exercise caution when using this tool. It's hasty implementation
is a result of time-limitations associated with a deadline.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class MetricsExtractor:
    """
    Extracts AUC and F1 scores from medical classification text files
    and performs statistical analysis to compare different models.
    """

    def __init__(self, directory="."):
        """
        Initialize the metrics extractor with a directory to search for txt files.

        Args:
            directory (str): Directory path containing metric text files
        """
        self.directory = directory
        self.metrics_data = {"standard": {}, "mm": {}}
        self.class_names = []

    def extract_metrics_from_file(self, filepath):
        """
        Extract AUC and F1 scores from a single metrics file.

        Args:
            filepath (str): Path to the metrics text file

        Returns:
            dict: Dictionary containing extracted metrics
        """
        try:
            with open(filepath, "r") as file:
                content = file.read()

            filename = os.path.basename(filepath).replace(".txt", "")

            metrics = {
                "filename": filename,
                "individual_class_auc": {},
                "individual_class_f1": {},
            }

            weighted_auc_match = re.search(r"Weighted Average AUC: ([\d\.]+)", content)
            macro_auc_match = re.search(r"Macro Average AUC: ([\d\.]+)", content)
            micro_auc_match = re.search(r"Micro Average AUC: ([\d\.]+)", content)

            if weighted_auc_match:
                metrics["weighted_auc"] = float(weighted_auc_match.group(1))
            if macro_auc_match:
                metrics["macro_auc"] = float(macro_auc_match.group(1))
            if micro_auc_match:
                metrics["micro_auc"] = float(micro_auc_match.group(1))

            for class_match in re.finditer(
                r"(\w+(?:\s\w+)*):\s*\n\s*AUC Score: ([\d\.]+)", content
            ):
                class_name = class_match.group(1).strip()
                auc_score = float(class_match.group(2))
                metrics["individual_class_auc"][class_name] = auc_score
                if class_name not in self.class_names:
                    self.class_names.append(class_name)

            f1_pattern = (
                r"Class: (\w+(?:\s\w+)*)\n\s*Precision: "
                r"[\d\.]+\n\s*Recall: [\d\.]+\n\s*F1-score: ([\d\.]+)"
            )
            for f1_match in re.finditer(f1_pattern, content):
                class_name = f1_match.group(1).strip()
                f1_score = float(f1_match.group(2))
                if class_name not in [
                    "micro avg",
                    "macro avg",
                    "weighted avg",
                    "samples avg",
                ]:
                    metrics["individual_class_f1"][class_name] = f1_score

            for avg_type in ["micro avg", "macro avg", "weighted avg"]:
                pattern = (
                    rf"Class: {avg_type}\n\s*Precision: "
                    r"[\d\.]+\n\s*Recall: [\d\.]+\n\s*F1-score: ([\d\.]+)"
                )
                match = re.search(pattern, content)
                if match:
                    clean_name = avg_type.replace(" ", "_")
                    metrics[f"{clean_name}_f1"] = float(match.group(1))

            return metrics

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None

    def process_all_files(self):
        """
        Process all txt files in the directory and categorize them by whether
        they contain '_mm_' in the filename or not, excluding anything with
        embd_ in the front since our statistical analysis will excluded
        embedded model results as we are not running tests on embedded + mm formally
        at this point.
        """
        txt_files = glob.glob(os.path.join(f"{self.directory}/results/tuning", "*.txt"))

        if not txt_files:
            print(f"No txt files found in directory: {self.directory}")
            return False

        print(f"Found {len(txt_files)} txt files to process")

        for filepath in txt_files:
            metrics = self.extract_metrics_from_file(filepath)
            if not metrics:
                continue

            filename = metrics["filename"]
            if "embd_" in filename:
                pass
            elif "_mm_" in filename:
                self.metrics_data["mm"][filename] = metrics
            else:
                self.metrics_data["standard"][filename] = metrics

        print(
            f"Processed {len(self.metrics_data['mm'])} MM files and "
            f"{len(self.metrics_data['standard'])} standard files"
        )

        return True

    def create_combined_dataframe(self):
        """
        Create a pandas DataFrame with all metrics from all files.

        Returns:
            pd.DataFrame: DataFrame containing all metrics
        """
        all_data = []

        for filename, metrics in self.metrics_data["standard"].items():
            row = {
                "filename": filename,
                "model_type": "standard",
                "weighted_auc": metrics.get("weighted_auc"),
                "macro_auc": metrics.get("macro_auc"),
                "micro_auc": metrics.get("micro_auc"),
                "micro_avg_f1": metrics.get("micro_avg_f1"),
                "macro_avg_f1": metrics.get("macro_avg_f1"),
                "weighted_avg_f1": metrics.get("weighted_avg_f1"),
            }

            for class_name in self.class_names:
                row[f"{class_name}_auc"] = metrics["individual_class_auc"].get(
                    class_name
                )
                row[f"{class_name}_f1"] = metrics["individual_class_f1"].get(class_name)

            all_data.append(row)

        for filename, metrics in self.metrics_data["mm"].items():
            row = {
                "filename": filename,
                "model_type": "mm",
                "weighted_auc": metrics.get("weighted_auc"),
                "macro_auc": metrics.get("macro_auc"),
                "micro_auc": metrics.get("micro_auc"),
                "micro_avg_f1": metrics.get("micro_avg_f1"),
                "macro_avg_f1": metrics.get("macro_avg_f1"),
                "weighted_avg_f1": metrics.get("weighted_avg_f1"),
            }

            for class_name in self.class_names:
                row[f"{class_name}_auc"] = metrics["individual_class_auc"].get(
                    class_name
                )
                row[f"{class_name}_f1"] = metrics["individual_class_f1"].get(class_name)

            all_data.append(row)

        return pd.DataFrame(all_data)

    def perform_paired_ttest(self):
        """
        Perform paired t-tests to compare standard and MM models.

        Note: This assumes there are matching pairs of files with and without "_mm_".

        Returns:
            dict: Dictionary containing t-test results
        """
        if not self.metrics_data["standard"] or not self.metrics_data["mm"]:
            print("Cannot perform paired t-test: missing data from one or both models")
            return None

        standard_base_names = set()
        mm_base_names = set()

        for filename in self.metrics_data["standard"].keys():
            standard_base_names.add(filename)

        for filename in self.metrics_data["mm"].keys():
            base_name = filename.replace("_mm_", "_")
            mm_base_names.add(base_name)

        common_base_names = []
        mm_to_standard_map = {}

        for std_name in standard_base_names:
            for mm_name in self.metrics_data["mm"].keys():
                mm_base = mm_name.replace("_mm_", "_")
                if std_name == mm_base or std_name in mm_base or mm_base in std_name:
                    common_base_names.append(std_name)
                    mm_to_standard_map[mm_name] = std_name
                    break

        if not common_base_names:
            print("No matching pairs found for t-test")
            return None

        print(f"Found {len(common_base_names)} matching pairs for t-test")

        metrics_to_test = [
            "weighted_auc",
            "macro_auc",
            "micro_auc",
            "micro_avg_f1",
            "macro_avg_f1",
            "weighted_avg_f1",
        ]

        for class_name in self.class_names:
            metrics_to_test.append(f"{class_name}_auc")
            metrics_to_test.append(f"{class_name}_f1")

        results = {}

        for metric in metrics_to_test:
            standard_values = []
            mm_values = []

            for mm_name, std_name in mm_to_standard_map.items():
                if (
                    mm_name in self.metrics_data["mm"]
                    and std_name in self.metrics_data["standard"]
                ):
                    mm_value = self._get_nested_metric(
                        self.metrics_data["mm"][mm_name], metric
                    )
                    std_value = self._get_nested_metric(
                        self.metrics_data["standard"][std_name], metric
                    )

                    if mm_value is not None and std_value is not None:
                        mm_values.append(mm_value)
                        standard_values.append(std_value)

            if (
                len(mm_values) > 1
                and len(standard_values) > 1
                and len(mm_values) == len(standard_values)
            ):
                t_stat, p_value = stats.ttest_rel(mm_values, standard_values)

                results[metric] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "mm_mean": np.mean(mm_values),
                    "standard_mean": np.mean(standard_values),
                    "mm_std": np.std(mm_values),
                    "standard_std": np.std(standard_values),
                    "difference": np.mean(mm_values) - np.mean(standard_values),
                    "significant": p_value < 0.05,
                    "sample_size": len(mm_values),
                }

        return results

    def _get_nested_metric(self, metrics_dict, metric_name):
        """
        Helper method to get metrics that might be nested inside dictionaries.

        Args:
            metrics_dict (dict): The metrics dictionary
            metric_name (str): The metric to retrieve

        Returns:
            float or None: The metric value or None if not found
        """
        if "_auc" in metric_name and not metric_name.startswith(
            ("weighted", "macro", "micro")
        ):
            class_name = metric_name.replace("_auc", "")
            return metrics_dict.get("individual_class_auc", {}).get(class_name)
        elif "_f1" in metric_name and not metric_name.startswith(
            ("micro_avg", "macro_avg", "weighted_avg")
        ):
            class_name = metric_name.replace("_f1", "")
            return metrics_dict.get("individual_class_f1", {}).get(class_name)
        else:
            return metrics_dict.get(metric_name)

    def generate_summary(self, ttest_results=None):
        """
        Generate a text summary of the results.

        Args:
            ttest_results (dict): Results from the t-test analysis

        Returns:
            str: Summary text
        """
        if not ttest_results:
            ttest_results = self.perform_paired_ttest()

        if not ttest_results:
            return "Insufficient data for analysis."

        summary = []
        summary.append("====== SUMMARY OF STATISTICAL ANALYSIS ======")
        summary.append(
            f"Number of standard files: {len(self.metrics_data['standard'])}"
        )
        summary.append(f"Number of MM files: {len(self.metrics_data['mm'])}")
        summary.append("")

        categories = {
            "Overall AUC Scores": ["weighted_auc", "macro_auc", "micro_auc"],
            "Overall F1 Scores": ["micro_avg_f1", "macro_avg_f1", "weighted_avg_f1"],
            "Individual Class AUC Scores": [
                m
                for m in ttest_results.keys()
                if "_auc" in m and not m.startswith(("weighted", "macro", "micro"))
            ],
            "Individual Class F1 Scores": [
                m
                for m in ttest_results.keys()
                if "_f1" in m
                and not m.startswith(("micro_avg", "macro_avg", "weighted_avg"))
            ],
        }

        for category, metrics in categories.items():
            if not any(m in ttest_results for m in metrics):
                continue

            summary.append(f"------ {category} ------")
            for metric in metrics:
                if metric not in ttest_results:
                    continue

                result = ttest_results[metric]
                metric_display = metric.replace("_", " ").title()

                mm_mean = result["mm_mean"]
                std_mean = result["standard_mean"]
                diff = result["difference"]
                p_value = result["p_value"]

                # Format the output
                summary.append(f"{metric_display}:")
                summary.append(
                    f"  MM model mean: {mm_mean:.4f} ± {result['mm_std']:.4f}"
                )
                summary.append(
                    f"  Standard model mean: {std_mean:.4f} ± "
                    f"{result['standard_std']:.4f}"
                )
                summary.append(
                    f"  Difference: {diff:.4f} "
                    f"({'+' if diff > 0 else ''}{diff/std_mean*100:.2f}%)"
                )
                summary.append(
                    f"  P-value: {p_value:.4f} "
                    f"({'Significant' if p_value < 0.05 else 'Not significant'})"
                )
                summary.append(f"  Sample size: {result['sample_size']}")
                summary.append("")

        return "\n".join(summary)

    def visualize_results(self, ttest_results=None, output_dir=None):
        """
        Generate visualizations of the results.

        Args:
            ttest_results (dict): Results from the t-test analysis
            output_dir (str): Directory to save visualizations

        Returns:
            None
        """
        if not ttest_results:
            ttest_results = self.perform_paired_ttest()

        if not ttest_results:
            print("Insufficient data for visualization.")
            return

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Separate AUC and F1 metrics
        auc_metrics = [
            "weighted_auc",
            "micro_auc",
        ]
        f1_metrics = [
            "micro_avg_f1",
            "weighted_avg_f1",
        ]

        # Create side-by-side plots for AUC and F1 scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot AUC metrics
        labels_auc = [m.replace("_", " ").title() for m in auc_metrics]
        mm_means_auc = [ttest_results.get(m, {}).get("mm_mean", 0) for m in auc_metrics]
        std_means_auc = [
            ttest_results.get(m, {}).get("standard_mean", 0) for m in auc_metrics
        ]
        mm_std_auc = [ttest_results.get(m, {}).get("mm_std", 0) for m in auc_metrics]
        std_std_auc = [
            ttest_results.get(m, {}).get("standard_std", 0) for m in auc_metrics
        ]

        x_auc = np.arange(len(labels_auc))
        width = 0.35

        ax1.bar(
            x_auc - width / 2,
            mm_means_auc,
            width,
            yerr=mm_std_auc,
            capsize=5,
            label="MM Model",
        )
        ax1.bar(
            x_auc + width / 2,
            std_means_auc,
            width,
            yerr=std_std_auc,
            capsize=5,
            label="Standard Model",
        )
        ax1.set_xlabel("Metric")
        ax1.set_ylabel("AUC Score (log scale)")
        ax1.set_yscale("log")
        ax1.set_title("AUC Metrics Comparison")
        ax1.set_xticks(x_auc)
        ax1.set_xticklabels(labels_auc, rotation=45, ha="right")
        ax1.legend()

        # Plot F1 metrics
        labels_f1 = [m.replace("_", " ").title() for m in f1_metrics]
        mm_means_f1 = [ttest_results.get(m, {}).get("mm_mean", 0) for m in f1_metrics]
        std_means_f1 = [
            ttest_results.get(m, {}).get("standard_mean", 0) for m in f1_metrics
        ]
        mm_std_f1 = [ttest_results.get(m, {}).get("mm_std", 0) for m in f1_metrics]
        std_std_f1 = [
            ttest_results.get(m, {}).get("standard_std", 0) for m in f1_metrics
        ]

        x_f1 = np.arange(len(labels_f1))

        ax2.bar(
            x_f1 - width / 2,
            mm_means_f1,
            width,
            yerr=mm_std_f1,
            capsize=5,
            label="MM Model",
        )
        ax2.bar(
            x_f1 + width / 2,
            std_means_f1,
            width,
            yerr=std_std_f1,
            capsize=5,
            label="Standard Model",
        )
        ax2.set_xlabel("Metric")
        ax2.set_ylabel("F1 Score (log scale)")
        ax2.set_yscale("log")
        ax2.set_title("F1 Metrics Comparison")
        ax2.set_xticks(x_f1)
        ax2.set_xticklabels(labels_f1, rotation=45, ha="right")
        ax2.legend()

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        plt.close()

        class_auc_metrics = [
            m
            for m in ttest_results.keys()
            if "_auc" in m and not m.startswith(("weighted", "macro", "micro"))
        ]

        if class_auc_metrics:
            class_auc_metrics.sort(
                key=lambda x: abs(ttest_results[x]["difference"]), reverse=True
            )

            labels = [
                m.replace("_auc", "").replace("_", " ").title()
                for m in class_auc_metrics
            ]
            mm_means = [ttest_results[m]["mm_mean"] for m in class_auc_metrics]
            std_means = [ttest_results[m]["standard_mean"] for m in class_auc_metrics]
            mm_stds = [ttest_results[m]["mm_std"] for m in class_auc_metrics]
            std_stds = [ttest_results[m]["standard_std"] for m in class_auc_metrics]

            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(labels))
            width = 0.35

            # Use log scale for class AUC scores
            ax.bar(
                x - width / 2,
                mm_means,
                width,
                yerr=mm_stds,
                capsize=5,
                label="MM Model",
            )
            ax.bar(
                x + width / 2,
                std_means,
                width,
                yerr=std_stds,
                capsize=5,
                label="Standard Model",
            )

            ax.set_xlabel("Class")
            ax.set_ylabel("AUC Score (log scale)")
            ax.set_yscale("log")
            ax.set_title("Comparison of Individual Class AUC Scores")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()

            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, "class_auc_comparison.png"))
            plt.close()

        significant_metrics = {
            m: ttest_results[m]["p_value"]
            for m in ttest_results
            if ttest_results[m]["p_value"] < 0.05
        }

        if significant_metrics:
            sorted_metrics = sorted(significant_metrics.items(), key=lambda x: x[1])
            labels = [m[0].replace("_", " ").title() for m in sorted_metrics]
            p_values = [m[1] for m in sorted_metrics]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(labels, p_values)

            ax.set_xlabel("P-value")
            ax.set_ylabel("Metric")
            ax.set_title("Statistically Significant Differences (p < 0.05)")
            ax.set_xlim(0, 0.05)
            ax.axvline(x=0.01, color="r", linestyle="--", label="p=0.01")
            ax.legend()

            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, "significant_differences.png"))
            plt.close()


def run_statistical_analysis():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract metrics from text files and perform analysis"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing text files with metrics",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/metrics_analysis_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        help="Directory to save visualizations",
        default="artifacts/metrics_analysis_results",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    extractor = MetricsExtractor(args.dir)
    if not extractor.process_all_files():
        print("No files were processed. Exiting.")
        return

    ttest_results = extractor.perform_paired_ttest()

    if ttest_results:
        summary = extractor.generate_summary(ttest_results)
        summary_path = os.path.join(args.output, "analysis_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Analysis summary saved to {summary_path}")

        df = extractor.create_combined_dataframe()
        csv_path = os.path.join(args.output, "metrics_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics data saved to {csv_path}")

        if args.visualize:
            viz_dir = os.path.join(args.output, "visualizations")
            extractor.visualize_results(ttest_results, viz_dir)
            print(f"Visualizations saved to {viz_dir}")
    else:
        print("Could not perform statistical analysis due to insufficient data.")

    print("Analysis complete.")


if __name__ == "__main__":
    run_statistical_analysis()
