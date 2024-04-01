import os
import torch
import numpy as np
import json
import concurrent
from mteb import MTEB
from datetime import datetime
import matplotlib.pyplot as plt
from launchkit.logging import logger
from openai import OpenAI
from anthropic import Anthropic

from adapt_embed.prompts import synthetic_data

import logging
file_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


openai_client = None
anthropic_client = None
def get_client(llm_type="openai"):
    global openai_client, anthropic_client
    if llm_type == "openai":
        if openai_client is None:
            openai_client = OpenAI()
        client = openai_client
    elif llm_type == "anthropic":
        if anthropic_client is None:
            anthropic_client = Anthropic()
        client = anthropic_client
    return client

def get_proj_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_device():
    # return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() 
                        else 'mps' if torch.backends.mps.is_available() 
                        else 'cpu')

def plot_comparison(results_with_names, exp_name="", variant=None, split='test', save=True, show=False, save_dir=None):
    """
    Plots a comparison graph for an arbitrary number of results dictionaries.
    
    results_with_names: Iterable of tuples of (results_dict, name) for each dataset to be plotted.
    exp_name: Experiment name.
    variant: dict of hyperparameters for the experiment.
    split: Data split to use for plotting (default is 'test').
    save: Whether to save the generated plots.
    show: Whether to show the generated plots.
    save_dir: Directory to save the plots. If None, a default directory structure will be used.
    """
    variant = variant or {}
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
    if len(task) != 1:
        file_logger.warn(f"All results dictionaries should be from the same task. Tasks found: {task}. One of these tasks will be chosen at random.")
    task = task.pop()
    file_logger.info(f"Analyzing task {task}.")
    
    n_groups = len(data_sets[0][0][list(data_sets[0][0].keys())[0]])  # Number of groups (1, 3, 5, 10, 100, 1000)
    n_datasets = len(data_sets)  # Number of datasets to compare
    spacing = 0.2  # Spacing between groups of bars
    group_width = 0.8  # Total width of each group of bars
    bar_width = group_width / n_datasets  # Width of each individual bar
    index = np.arange(n_groups) * (group_width + spacing)

    # Plot each metric for all datasets
    for metric in data_sets[0][0].keys():
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (data, name) in enumerate(data_sets):
            k_order = sorted(data[metric].keys(), key=lambda x: int(x.split('_at_')[1]))
            bars = ax.bar(index + i * bar_width, [data[metric][k] for k in k_order], bar_width, label=name)
        
        ax.set_xlabel('@k')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()}{" on " + exp_name if exp_name else ""}')
        ax.set_xticks(index + group_width / 2)
        ax.set_xticklabels([k.split('_at_')[1] for k in k_order])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        fig.subplots_adjust(right=0.8)

        if save:
            snapshot_dir = logger.get_snapshot_dir()
            if snapshot_dir:
                default_save_dir = os.path.join(snapshot_dir, 'imgs')
            else:
                default_save_dir = os.path.join(get_proj_dir(), 'results', exp_name or 'default', variant.get('model_name', 'default'), task, 'imgs',
                                                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            save_dir_actual = save_dir or default_save_dir
            os.makedirs(save_dir_actual, exist_ok=True)
            plt.savefig(f"{save_dir_actual}/{metric}.png")

        if show:
            plt.show()
        plt.close()

def stringify_corpus_item(item: dict | str, sep='\n') -> str:
    if isinstance(item, dict):
        return (item['title'] + sep + item['text']).strip() if 'title' in item else item['text'].strip()
    else:
        return item.strip()
        
def gen_synthetic_data(query, n, examples=None, llm="gpt-4-turbo-preview"):
    documents = []

    query_messages = synthetic_data.get_messages(query, examples=examples)
    if llm.startswith("gpt"):
        llm_type = "openai"
        openai_client = get_client(llm_type)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(openai_client.chat.completions.create,
                                    model=llm,
                                    messages=query_messages)
                        for _ in range(n)]
            for future in concurrent.futures.as_completed(futures):
                completion = future.result()
                document = completion.choices[0].message.content
                documents.append(document)
    elif llm.startswith("claude"):
        llm_type = "anthropic"
        anthropic_client = get_client(llm_type)
        system_message = query_messages.pop(0)['content']
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(anthropic_client.messages.create,
                                       model=llm,
                                       max_tokens=200,
                                       messages=query_messages,
                                       system=system_message)
                       for _ in range(n)]
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                document = response.content[0].text
                breakpoint()
                documents.append(document)
    else:
        raise ValueError(f"Unsupported LLM model: {llm}")
    return documents

def get_mteb_results(task, results_file, model=None):
    if os.path.exists(results_file):
        with open(results_file) as json_file:
            results = {task: json.load(json_file)}
    elif model is not None:
        results = MTEB(tasks=[task]).run(model, output_folder=os.path.dirname(results_file))
    else:
        raise ValueError("Either model or existing results_file must be provided.")
    return results

class LocalLogger:
    """
    Context manager to temporarily change logger's snapshot directory to os.path.join(get_snapshot_directory(), VALUE)
    and add KEY: VALUE to variant in-place. Restores original snapshot directory and variant value on exit.
    """
    def __init__(self, key, value, variant):
        self.key = key
        self.snapshot_prefix = value
        self.snapshot_dir = None
        self.sub_dir = None
        self.original_tabular_outputs = None
        self.original_tabular_fds = None
        self.original_tabular_header_written = None
        self.tabular_file = None
        self.variant = variant
        self.original_variant_value = None

    def __enter__(self):
        self.snapshot_dir = logger.get_snapshot_dir()
        self.sub_dir = os.path.join(self.snapshot_dir, self.snapshot_prefix)
        os.makedirs(self.sub_dir, exist_ok=True)
        logger.set_snapshot_dir(self.sub_dir)

        # Save the original tabular outputs and create new ones in the sub directory
        self.original_tabular_outputs = logger._tabular_outputs
        self.original_tabular_fds = logger._tabular_fds
        self.original_tabular_header_written = logger._tabular_header_written
        
        self.original_variant_value = self.variant.get(self.key, None)
        self.variant[self.key] = self.snapshot_prefix
        logger.log_variant(os.path.join(self.sub_dir, 'variant.json'), self.variant)

        logger._tabular_outputs = []
        logger._tabular_fds = {}
        logger._tabular_header_written = set()

        self.tabular_file = os.path.join(self.sub_dir, 'progress.csv')
        logger._tabular_outputs.append(self.tabular_file)
        logger._tabular_fds[self.tabular_file] = open(self.tabular_file, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the tabular file descriptor in the sub directory
        logger._tabular_fds[self.tabular_file].close()

        # Restore the original tabular outputs, file descriptors, and header written state
        logger._tabular_outputs = self.original_tabular_outputs
        logger._tabular_fds = self.original_tabular_fds
        logger._tabular_header_written = self.original_tabular_header_written

        # Restore the original snapshot directory and variant
        logger.set_snapshot_dir(self.snapshot_dir)
        if self.original_variant_value is None:
            self.variant.pop(self.key)
        else:
            self.variant[self.key] = self.original_variant_value