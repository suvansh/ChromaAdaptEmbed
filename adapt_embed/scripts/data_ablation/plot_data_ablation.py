import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def plot_data_ablation(directory, task_name, split_name, adapter_types, num_queries=None, baseline_path=None, save_dir=None):
    assert os.path.exists(directory)
    data = {s: {} for s in adapter_types}
    metric_k_values = {}
    baseline_data = {}
    suffix = "_baseline" if baseline_path else ""
    save_dir = save_dir or os.path.join(directory, f"data_ablation{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load baseline data if provided
    if baseline_path:
        with open(baseline_path, 'r') as file:
            baseline_metrics = json.load(file)
            if split_name in baseline_metrics:
                baseline_data = baseline_metrics[split_name]

    for subdir, dirs, files in os.walk(directory):
        if 'variant.json' in files:
            with open(os.path.join(subdir, 'variant.json'), 'r') as file:
                variant_data = json.load(file)
                data_subset_frac = variant_data.get('data_subset_frac', 1.0)
                if num_queries is not None:
                    data_subset_frac *= num_queries
            
            for s in adapter_types:
                task_file_path = os.path.join(subdir, s, f"{task_name}.json")
                if os.path.exists(task_file_path):
                    with open(task_file_path, 'r') as file:
                        results = json.load(file)
                        if split_name in results:
                            metrics = results[split_name]
                            for key, value in metrics.items():
                                if '_at_' in key:
                                    metric_name, k = key.split('_at_')
                                    k = int(k)
                                    if metric_name not in data[s]:
                                        data[s][metric_name] = defaultdict(list)
                                        metric_k_values[metric_name] = set()
                                    metric_k_values[metric_name].add(k)
                                    data[s][metric_name][k].append((data_subset_frac, value))
    
    # Plotting
    for s in adapter_types:
        os.makedirs(os.path.join(save_dir, s), exist_ok=True)
        for metric, points in data[s].items():
            plt.figure(figsize=(10, 5))
            for k in sorted(metric_k_values[metric]):
                if k in points:
                    fracs, vals = zip(*sorted(points[k]))
                    line, = plt.plot(fracs, vals, label=f'k={k}', marker='o')
                    # Plot baseline if available
                    baseline_key = f'{metric}_at_{k}'
                    if baseline_key in baseline_data:
                        plt.axhline(y=baseline_data[baseline_key], color=line.get_color(), linestyle='--')

            plt.title(f'{metric.upper()} for {s} by data subset')
            plt.xlabel('Data Subset Fraction' if num_queries is None else '# of Query-Document annotations (excluding negative sampling)')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, s, f'{metric}.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot metrics from experimental settings")
    parser.add_argument("task_name", type=str, help="Name of the task (stats filename without .json)")
    parser.add_argument("directory", type=str, help="The directory containing experiment subdirectories")
    parser.add_argument("adapter_types", type=str, nargs='+', help="List of strings representing adapter types (sub-subdirectories)")
    parser.add_argument("--num-queries", "-n", type=int, help="Number of queries to plot (optional)")
    parser.add_argument("--split-name", "-s", type=str, default="test", help="The split name within the task performance JSON file")
    parser.add_argument("--baseline", "-b", type=str, help="Path to the baseline performance JSON file (optional)")
    parser.add_argument("--save-dir", "-d", type=str, default=None, help="Path to save the plots (optional)")
    
    args = parser.parse_args()
    plot_data_ablation(args.directory, args.task_name, args.split_name, args.adapter_types, args.num_queries, args.baseline, args.save_dir)

if __name__ == "__main__":
    main()
