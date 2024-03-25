import os
import json
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader

from adapt_embed.datasets.triplet import TripletDataset
from adapt_embed.datasets.inputexample import InputExampleDataset
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device, get_mteb_results

from launchkit import launcher_util
from launchkit.sweeper import DeterministicHyperparameterSweeper
from launchkit.logging import logger


proj_dir = get_proj_dir()
device = get_device()
exp_name = "finetune"


def run_experiment(variant):
    model_name = variant['model_name']
    task = variant['task']
    split = variant['split']
    num_epochs = variant['num_epochs']
    batch_size = variant['batch_size']
    lr = variant['lr']
    loss_type = variant['loss_type']
    results_files = variant.get('results_files', [])
    results_every = variant.get('results_every', 0)

    model = SentenceTransformer(model_name, device=device)
    finetuned_model = SentenceTransformer(model_name, device=device)
    torch_loss = {'mse': torch.nn.MSELoss(), 'bce': torch.nn.BCELoss()}.get(loss_type, None)
    if torch_loss is None:
        raise ValueError(f"Unknown loss type: {loss_type}. Must be one of 'mse' or 'bce'.")

    dataset = None
    def get_dataset():
        nonlocal dataset
        if dataset is None:
            dataset = InputExampleDataset(TripletDataset(MTEB(tasks=[task]).tasks[0], negative_sampling=True, split=split, relevance_threshold=0.5))
        return dataset
    
    def get_results(model, task):
        return get_mteb_results(task, os.path.join(logger.get_snapshot_dir(), 'results.json'), model=model)

    def train_and_evaluate(model, save_dir, results_every=0):
        """
        Trains model and saves it to save_dir. Evaluates every results_every epochs. Default is 0 (only at the end).
        """
        os.makedirs(save_dir, exist_ok=True)
        print("Finetuning the model...")
        train_data = DataLoader(get_dataset(), shuffle=True, batch_size=batch_size)
        valid_data = zip(*[(*dataset[i].texts, dataset[i].label) for i in range(0, 2501, 5)])  # 500 pairs, offset is odd to get pos and neg labels since they alternate
        evaluator = evaluation.EmbeddingSimilarityEvaluator(*valid_data, name=task)
        loss_fn = losses.CosineSimilarityLoss(model, loss_fct=torch_loss)
        evaluation_steps = len(train_data) * max(0, results_every)
        evaluation_dicts = []
        callback = lambda score, epoch, steps: evaluation_dicts.append({'epoch': epoch, 'loss': score, 'steps': steps})
        model.fit(
            [(train_data, loss_fn)],
            epochs=num_epochs, output_path=save_dir,
            evaluator=evaluator, evaluation_steps=evaluation_steps,
            callback=callback,
            optimizer_params={'lr': lr}
        )
        results = get_results(model, task)
        logger.record_dict(**evaluation_dicts[-1], **results[task][split])
        logger.dump_tabular()
        for eval_dict in evaluation_dicts[:-1]:
            logger.record_dict(**eval_dict)
            logger.dump_tabular()
        return results
    
    save_dir = os.path.join(logger.get_snapshot_dir(), 'finetuned_model')
    results_finetune = train_and_evaluate(finetuned_model, save_dir, results_every=results_every)

    external_results = {name: {task: json.load(open(results_file))} for results_file, name in results_files if os.path.exists(results_file)}

    baseline_results = get_mteb_results(task, os.path.join(proj_dir, 'results', model_name, f"{task}.json"), model=model)
    plot_comparison([(baseline_results, "Baseline"),
                     (results_finetune, "Finetuned"),
                     *[(results, name) for name, results in external_results.items()]],
                    exp_name, variant)

if __name__ == "__main__":
    variants = dict(
        model_name=["all-MiniLM-L6-v2"],
        task=['CQADupstackEnglishRetrieval'],
        split=['test'],
        num_epochs=[15],
        batch_size=[256],
        lr=[0, 1e-5, 3e-5, 1e-4],
        loss_type=['mse', 'bce']
    )
    results_files = []

    search_space = {}
    sweeper = DeterministicHyperparameterSweeper(variants)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['results_files'] = results_files
        variant['results_every'] = 3
        launcher_util.run_experiment(
            run_experiment,
            variant=variant,
            exp_prefix='finetune',
            mode='local',
            snapshot_mode='last',
            base_log_dir=os.path.join(proj_dir, 'results', 'logs')
        )