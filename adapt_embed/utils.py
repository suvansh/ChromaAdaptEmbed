import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt


def get_proj_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() 
                        else 'mps' if torch.backends.mps.is_available() 
                        else 'cpu')

def plot_comparison(results_with_names, exp_name, split='test', save=True, show=False, save_dir=None):
    """
    Plots a comparison graph for an arbitrary number of results dictionaries.
    
    results_with_names: Iterable of tuples of (results_dict, name) for each dataset to be plotted.
    exp_name: Experiment name.
    split: Data split to use for plotting (default is 'test').
    save: Whether to save the generated plots.
    show: Whether to show the generated plots.
    save_dir: Directory to save the plots. If None, a default directory structure will be used.
    """
    # Preprocess data for each results dictionary
    data_sets = []
    task = set()
    for results, name in results_with_names:
        task.add(list(results.keys())[0])
        metrics = list(results.values())[0][split]
        data = {
            metric.split('_at_')[0]: {k: float(v) for k, v in metrics.items() if k.startswith(metric.split('_at_')[0])}
            for metric in metrics if '_at_' in metric
        }
        data_sets.append((data, name))
    assert len(task) == 1, f"All results dictionaries must be from the same task. Tasks found: {task}"
    task = task.pop()
    
    n_groups = len(data_sets[0][0][list(data_sets[0][0].keys())[0]])  # Number of groups (1, 3, 5, 10, 100, 1000)
    index = np.arange(n_groups) * 1.5  # Group positions
    bar_width = 0.35  # Width of the bars

    # Plot each metric for all datasets
    for metric in data_sets[0][0].keys():
        fig, ax = plt.subplots()
        for i, (data, name) in enumerate(data_sets):
            k_order = sorted(data[metric].keys(), key=lambda x: int(x.split('_at_')[1]))
            bars = ax.bar(index + i * bar_width, [data[metric][k] for k in k_order], bar_width, label=name)

        ax.set_xlabel('@k')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} on {exp_name}')
        ax.set_xticks(index + bar_width * len(data_sets) / 2)
        ax.set_xticklabels([k.split('_at_')[1] for k in k_order])
        ax.legend()

        if save:
            save_dir_actual = save_dir or f'{get_proj_dir()}/results/{exp_name}/{task}/imgs'
            os.makedirs(save_dir_actual, exist_ok=True)
            plt.savefig(f"{save_dir_actual}/{metric}.png")

        if show:
            plt.show()

def stringify_corpus_item(item: dict | str, sep='\n') -> str:
        if isinstance(item, dict):
            return (item['title'] + sep + item['text']).strip() if 'title' in item else item['text'].strip()
        else:
            return item.strip()