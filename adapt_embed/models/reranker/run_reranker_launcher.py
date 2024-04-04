import os
import json
import torch
import numpy as np
from mteb import MTEB
from sentence_transformers import SentenceTransformer

from adapt_embed.models.linear.linear import LinearAdapter
from adapt_embed.models.reranker.reranker import Reranker
from adapt_embed.datasets import PairwiseScoreDataset
from adapt_embed.datasets.inputexample import InputExampleDataset
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device, get_mteb_results

from launchkit import launcher_util
from launchkit.sweeper import DeterministicHyperparameterSweeper
from launchkit.logging import logger

proj_dir = get_proj_dir()
device = get_device()
exp_name = "reranker"

def run_experiment(variant):
    model_name = variant['model_name']
    rerank_model_name = variant['rerank_model_name']
    task = variant['task']
    split = variant['split']
    num_epochs = variant['num_epochs']
    batch_size = variant['batch_size']
    lr = variant['lr']
    results_files = variant.get('results_files', [])
    relevance_threshold = variant.get('relevance_threshold', 0.5)
    data_negative_sampling = variant.get('data_negative_sampling', True)
    data_synthetic_gen = variant.get('data_synthetic_gen', False)
    data_synthetic_data_path = variant.get('data_synthetic_data_path', None)
    data_augmentation_threshold = variant.get('data_augmentation_threshold', 10)
    data_llm = variant.get('data_llm', 'gpt-4-turbo-preview')

    model = SentenceTransformer(model_name, device=device)
    reranker_model = Reranker(rerank_model_name).to(device)
    reranker_baseline_model = Reranker(rerank_model_name).to(device)

    task_class = MTEB(tasks=[task]).tasks[0]
    # need to import after prev line to avoid error from search over subclasses
    from adapt_embed.eval.reranker_retrieval import RerankerRetrievalTask
    reranker_task_class = RerankerRetrievalTask.from_task(type(task_class))()
    reranker_task_class.load_data()

    dataset = InputExampleDataset(PairwiseScoreDataset(task_class, thresholded=True, relevance_threshold=relevance_threshold, split=split,
                                                       negative_sampling=data_negative_sampling,
                                                       synthetic_data=data_synthetic_gen,
                                                       synthetic_data_path=data_synthetic_data_path,
                                                       data_augmentation_threshold=data_augmentation_threshold,
                                                       llm=data_llm))

    reranker_weights_dir = os.path.join(logger.get_snapshot_dir(), 'reranker_model')
    reranker_model.fit(dataset, num_epochs=num_epochs, lr=lr, batch_size=batch_size, model_save_path=reranker_weights_dir)

    def get_results(exp_name, model, task):
        return get_mteb_results(task, os.path.join(logger.get_snapshot_dir(), f"{exp_name}_results.json"), model=model)

    results_reranked = get_results(exp_name, reranker_model, task)
    results_reranked_baseline = get_results(f"{exp_name}_baseline", reranker_baseline_model, task)

    external_results = {name: {task: json.load(open(results_file))} for results_file, name in results_files if os.path.exists(results_file)}

    baseline_results = get_mteb_results(task, os.path.join(proj_dir, 'results', model_name, f"{task}.json"), model=model)
    plot_comparison([(baseline_results, "Baseline"),
                     (results_reranked_baseline, "Pre-Trained Reranker"),
                     (results_reranked, "Fine-tuned Reranker"),
                     *[(results, name) for name, results in external_results.items()]],
                    exp_name, variant)

if __name__ == "__main__":
    """ hyperparameters """
    variants_list = [
        dict(
            exp_prefix=['reranker'],
            model_name=["all-MiniLM-L6-v2"],
            rerank_model_name=["distilroberta-base"],
            task=['CQADupstackEnglishRetrieval'],
            split=['test'],
            num_epochs=[10],
            batch_size=[64],
            lr=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
            data_negative_sampling=[True, False],
            data_synthetic_gen=[True, False],
            data_augmentation_threshold=[5],
            data_llm=['claude-3-sonnet-20240229']
        )
    ]
    results_files = []

    variants = [variant for variants in variants_list for variant in DeterministicHyperparameterSweeper(variants).iterate_hyperparameters()]

    """ run experiments """
    for exp_id, variant in enumerate(variants):
        variant['results_files'] = results_files
        launcher_util.run_experiment(
            run_experiment,
            variant=variant,
            exp_prefix=variant['exp_prefix'],
            mode='local',
            snapshot_mode='last',
            base_log_dir=os.path.join(proj_dir, 'results', 'logs')
        )