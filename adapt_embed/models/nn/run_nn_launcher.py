import os
import json
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer

from adapt_embed.datasets import TripletDataset, PairwiseScoreDataset
from adapt_embed.models.nn.nn import NNAdapter
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device, get_mteb_results, LocalLogger

from launchkit import launcher_util
from launchkit.sweeper import DeterministicHyperparameterSweeper
from launchkit.logging import logger


proj_dir = get_proj_dir()
device = get_device()
exp_name = "nn"

def run_experiment(variant):
    model_name = variant['model_name']
    task = variant['task']
    split = variant['split']
    eval_split = variant.get('eval_split', split)
    hidden_sizes = variant.get('hidden_sizes', [512])
    num_epochs = variant['num_epochs']
    lr = variant['lr']
    batch_size = variant['batch_size']
    triplet_margin = variant.get('triplet_margin', None)
    loss_type = variant['loss_type']
    data_negative_sampling = variant.get('data_negative_sampling', True)
    data_synthetic_gen = variant.get('data_synthetic_gen', False)
    data_synthetic_data_path = variant.get('data_synthetic_data_path', None)
    data_use_gold_data = variant.get('data_use_gold_data', True)
    data_augmentation_threshold = variant.get('data_augmentation_threshold', 10)
    data_subset_frac = variant.get('data_subset_frac', 1.0)
    data_llm = variant.get('data_llm', 'gpt-4-turbo-preview')

    if triplet_margin is None and loss_type == 'triplet':
        raise ValueError("Must provide triplet_margin if using triplet loss.")
    if loss_type not in ['triplet', 'mse', 'bce']:
        raise ValueError(f"Invalid loss type: {loss_type}")

    model = SentenceTransformer(model_name, device=device)
    for p in model.parameters():
        p.requires_grad = False

    dataset = None
    def get_dataset():
        nonlocal dataset
        if dataset is None:
            if loss_type == 'triplet':
                dataset = TripletDataset(MTEB(tasks=[task]).tasks[0],
                                         split=split, relevance_threshold=0.5,
                                         negative_sampling=data_negative_sampling,
                                         synthetic_data=data_synthetic_gen,
                                         synthetic_data_path=data_synthetic_data_path,
                                         use_gold_data=data_use_gold_data,
                                         data_augmentation_threshold=data_augmentation_threshold,
                                         llm=data_llm,
                                         eval_splits=[split, eval_split])
            else:
                dataset = PairwiseScoreDataset(MTEB(tasks=[task]).tasks[0],
                                               split=split, relevance_threshold=0.5, normalized=True, eps=1e-8,
                                               negative_sampling=data_negative_sampling,
                                               synthetic_data=data_synthetic_gen,
                                               synthetic_data_path=data_synthetic_data_path,
                                               use_gold_data=data_use_gold_data,
                                               data_augmentation_threshold=data_augmentation_threshold,
                                               llm=data_llm,
                                               eval_splits=[split, eval_split])
        return dataset
    
    def get_results(model, task, eval_split=eval_split):
        return get_mteb_results(task, os.path.join(logger.get_snapshot_dir(), 'results.json'), model=model, eval_splits=[eval_split])

    def train_and_evaluate(adapted_model, weights_file, adapter_type, force=False):
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        losses = []
        if os.path.exists(weights_file) and not force:
            adapted_model.load(weights_file)
        else:
            print(f"Training {adapter_type} NN Adapter...")
            losses = adapted_model.fit(get_dataset(), subset_frac=data_subset_frac, num_epochs=num_epochs, lr=lr, batch_size=batch_size, loss_type=loss_type, margin=triplet_margin, model_save_path=weights_file)
        results = get_results(adapted_model, task)
        # log last first so all the results keys are added
        logger.record_dict({'epoch': num_epochs-1, 'loss': losses[-1], **results[task][eval_split]})
        logger.dump_tabular()
        for i, loss in enumerate(losses[:-1]):  # log rest
            logger.record_dict({'epoch': i, 'loss': loss})
            logger.dump_tabular()
        return results


    results = {}

    with LocalLogger('adapter_type', 'query_adapted', variant):
        qa_weights_file = os.path.join(logger.get_snapshot_dir(), 'weights_query_adapted.pt')
        q_adapted_model = NNAdapter(model, model.get_sentence_embedding_dimension(), query_only=True, hidden_sizes=hidden_sizes).to(device)
        results['query_adapted'] = train_and_evaluate(q_adapted_model, qa_weights_file, 'Query-Adapted')

    with LocalLogger('adapter_type', 'adapted', variant):
        weights_file = os.path.join(logger.get_snapshot_dir(), 'weights.pt')
        adapted_model = NNAdapter(model, model.get_sentence_embedding_dimension(), hidden_sizes=hidden_sizes).to(device)
        results['adapted'] = train_and_evaluate(adapted_model, weights_file, 'Joint')

    # with LocalLogger('adapter_type', 'query_first', variant):
    #     query_first_weights_file = os.path.join(logger.get_snapshot_dir(), 'weights_query_first.pt')
    #     query_first_adapted_model = NNAdapter(model, model.get_sentence_embedding_dimension(), hidden_sizes=hidden_sizes).to(device)
    #     with torch.no_grad():
    #         query_first_adapted_model.model[0].weight.copy_(q_adapted_model.model[0].weight)
    #         query_first_adapted_model.model[0].bias.copy_(q_adapted_model.model[0].bias)
    #     results['query_first'] = train_and_evaluate(query_first_adapted_model, query_first_weights_file, 'Query-First')
        
    # with LocalLogger('adapter_type', 'separate', variant):
    #     separate_weights_file = os.path.join(logger.get_snapshot_dir(), 'weights_separate.pt')
    #     separate_adapted_model = NNAdapter(model, model.get_sentence_embedding_dimension(), separate_embeddings=True, hidden_sizes=hidden_sizes).to(device)
    #     results['separate'] = train_and_evaluate(separate_adapted_model, separate_weights_file, 'Separate')

    baseline_results = get_mteb_results(task, os.path.join(proj_dir, 'results', model_name, f"{task}.json"), model=model, eval_splits=[eval_split])
    plot_comparison([(baseline_results, "Baseline"),
                     (results['adapted'], "NN (Joint)"),
                     (results['query_adapted'], "NN (Query-Only)"),
                    #  (results['query_first'], "NN (Query-First)"),
                    #  (results['separate'], "NN (Separate)")
                    ],
                    exp_name, variant, split=eval_split)

if __name__ == "__main__":
    # tasks = ['ClimateFEVER', 'BSARDRetrieval']
    # tasks = ['DBPedia', 'HagridRetrieval']
    # tasks = ['MSMARCO', 'CQADupstackEnglishRetrieval', 'SpanishPassageRetrievalS2S']
    # tasks = ['QuoraRetrieval', 'SciFact', 'MSMARCO', 'QuoraPLRetrieval']
    # tasks = ['Ko-miracl']
    tasks = ['NFCorpus']
    variants_list = [
        # triplet
        dict(
            model_name=["all-MiniLM-L6-v2"],
            task=tasks,
            split=['test'],
            eval_split=['test'],
            num_epochs=[10],
            hidden_sizes=[[512], [128], [128, 128]],
            lr=[1e-2, 3e-3, 1e-3, 3e-4],
            batch_size=[512],
            triplet_margin=[0.3],
            loss_type=['triplet'],
            data_llm=['claude-3-sonnet-20240229'],
            data_augmentation_threshold=[5],
            data_synthetic_gen=[False],
            data_negative_sampling=[True]
        ),
        # pairwise
        dict(
            model_name=["all-MiniLM-L6-v2"],
            task=tasks,
            split=['test'],
            eval_split=['test'],
            num_epochs=[10],
            hidden_sizes=[[512], [128], [128, 128]],
            lr=[1e-2, 3e-3, 1e-3, 3e-4],
            batch_size=[512],
            loss_type=['mse'],
            data_llm=['claude-3-sonnet-20240229'],
            data_augmentation_threshold=[5],
            data_synthetic_gen=[False],
            data_negative_sampling=[True]
        )
    ]

    variants = [variant for variants in variants_list for variant in DeterministicHyperparameterSweeper(variants).iterate_hyperparameters()]

    for exp_id, variant in enumerate(variants):
        launcher_util.run_experiment(
            run_experiment,
            variant=variant,
            exp_prefix=f'nn-{variant["task"]}',
            mode='local',
            snapshot_mode='last',
            base_log_dir=os.path.join(proj_dir, 'results', 'logs')
        )
