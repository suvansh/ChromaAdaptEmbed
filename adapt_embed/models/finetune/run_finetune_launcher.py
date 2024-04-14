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
    results_files = variant.get('results_files', [])
    results_every = variant.get('results_every', 0)
    data_negative_sampling = variant.get('data_negative_sampling', True)
    data_synthetic_gen = variant.get('data_synthetic_gen', False)
    data_augmentation_threshold = variant.get('data_augmentation_threshold', 10)
    score_triplet = variant.get('score_triplet', False)
    loss_type = variant.get('loss_type', None)
    triplet_margin = variant.get('triplet_margin', None)

    model = SentenceTransformer(model_name, device=device)
    finetuned_model = SentenceTransformer(model_name, device=device)
    torch_loss = {'mse': torch.nn.MSELoss(), 'bce': torch.nn.BCELoss()}.get(loss_type, None)

    if not score_triplet and triplet_margin is None:
        raise ValueError("Must provide triplet_margin if not scoring triplet loss.")
    if score_triplet and torch_loss is None:
        raise ValueError("Must provide known loss_type if scoring triplet loss. Options: 'mse', 'bce'.")

    dataset = None
    def get_dataset():
        nonlocal dataset
        if dataset is None:
            dataset = InputExampleDataset(
                        TripletDataset(MTEB(tasks=[task]).tasks[0],
                                     split=split, relevance_threshold=0.5,
                                     negative_sampling=data_negative_sampling,
                                     synthetic_data=data_synthetic_gen,
                                     data_augmentation_threshold=data_augmentation_threshold
                            ), score_triplet=score_triplet
                        )
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
        if score_triplet:
            valid_data = zip(*[(*dataset[i].texts, dataset[i].label) for i in range(0, 2501, 5)])  # 500 pairs, offset is odd to get pos and neg labels since they alternate
            evaluator = evaluation.EmbeddingSimilarityEvaluator(*valid_data, name=task, batch_size=batch_size)
            loss_fn = losses.CosineSimilarityLoss(model, loss_fct=torch_loss, cos_score_transformation=lambda x: torch.clamp(x, 1e-8, 1-1e-8))
        else:
            valid_data = zip(*[dataset[i].texts for i in range(0, 2501, 5)])
            evaluator = evaluation.TripletEvaluator(*valid_data, name=task, batch_size=batch_size)
            loss_fn = losses.TripletLoss(model, triplet_margin=triplet_margin)
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
        logger.record_dict({**evaluation_dicts[-1], **results[task][split]})
        logger.dump_tabular()
        for eval_dict in evaluation_dicts[:-1]:
            logger.record_dict(eval_dict)
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
    tasks = ['QuoraRetrieval']
    tasks = ['SpanishPassageRetrievalS2S']
    tasks = ['Ko-miracl']
    """ hyperparameters """
    variants_list = [
        # triplet loss
        dict(
            model_name=["all-MiniLM-L6-v2"],
            task=tasks,
            split=['test'],
            data_augmentation_threshold=[5],
            num_epochs=[12],
            batch_size=[64],
            lr=[3e-3, 1e-3, 3e-4, 1e-4],
            score_triplet=[False],
            triplet_margin=[1/3]
        ),
        # converting triplets to classification and using loss on cosine similarity
        dict(
            model_name=["all-MiniLM-L6-v2"],
            task=tasks,
            split=['test'],
            data_augmentation_threshold=[5],
            data_negative_sampling=[True, False],
            num_epochs=[12],
            batch_size=[64],
            lr=[3e-3, 1e-3, 3e-4, 1e-4],
            score_triplet=[True],
            loss_type=['mse']
        )
    ]
    results_files = []
    results_every = 3

    variants = [variant for variants in variants_list for variant in DeterministicHyperparameterSweeper(variants).iterate_hyperparameters()]

    """ run experiments """
    for exp_id, variant in enumerate(variants):
        variant['results_files'] = results_files
        variant['results_every'] = results_every
        launcher_util.run_experiment(
            run_experiment,
            variant=variant,
            exp_prefix=f'finetune-{variant["task"]}',
            mode='local',
            snapshot_mode='last',
            base_log_dir=os.path.join(proj_dir, 'results', 'logs')
        )
