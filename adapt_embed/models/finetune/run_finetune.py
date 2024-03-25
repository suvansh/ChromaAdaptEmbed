# NOTE: deprecated in favor of run_finetune_launcher.py
import os
import json
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer, losses, evaluation
from torch.utils.data import DataLoader, Subset

from adapt_embed.datasets import TripletDataset
from adapt_embed.datasets.inputexample import InputExampleDataset
from adapt_embed.models.linear.linear import LinearAdapter
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device

proj_dir = get_proj_dir()
device = get_device()

""" What are we doing? """
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device=device)
exp_name = "finetune"
task = 'CQADupstackEnglishRetrieval'
# task = 'BSARDRetrieval'
split = 'test'
save_dir = f"{proj_dir}/results/{exp_name}/{task}/model"
retrain = True
if retrain:
    finetuned_model = SentenceTransformer(model_name, device=device)
else:
    finetuned_model = SentenceTransformer(save_dir, device=device)

dataset = None
def get_dataset():
    global dataset
    if dataset is None:
        dataset = InputExampleDataset(TripletDataset(MTEB(tasks=[task]).tasks[0], negative_sampling=True, split=split, relevance_threshold=0.5))
    return dataset

""" Load/train the model """
if retrain:
    num_epochs = 10
    batch_size = 128

    train_data = DataLoader(get_dataset(), shuffle=True, batch_size=batch_size)
    valid_data = zip(*[(*dataset[i].texts, dataset[i].label) for i in range(0, 2501, 5)])  # 500 pairs, offset is odd to get pos and neg labels since they alternate
    evaluator = evaluation.EmbeddingSimilarityEvaluator(*valid_data, name=task)
    loss_fn = losses.CosineSimilarityLoss(finetuned_model)
    finetuned_model.fit([(train_data, loss_fn)], epochs=num_epochs, output_path=save_dir,
                         evaluator=evaluator,
                        optimizer_params={'lr': 3e-2})

""" Load linear baselines """
weights_file = f"{proj_dir}/results/linear/{task}/weights.pt"
qa_weights_file = f"{proj_dir}/results/linear/{task}/qa_weights.pt"
linear_exists = False
if os.path.exists(weights_file) and os.path.exists(qa_weights_file):
    try:
        adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension()).to(device)
        adapted_model.load_state_dict(torch.load(weights_file))
        q_adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension(), query_only=True).to(device)
        q_adapted_model.load_state_dict(torch.load(qa_weights_file))
        linear_exists = True
    except:
        pass  # just use the finetuned model

""" Evaluate the model """
def get_results(exp_name, model, task):
    results_file = f"{proj_dir}/results/{exp_name}/{task}/{task}.json"
    if os.path.exists(results_file):
        with open(results_file) as json_file:
            return {task: json.load(json_file)}
    else:
        return MTEB(tasks=[task]).run(model, output_folder=os.path.dirname(results_file))

results = get_results("embed_baseline", model, task)
results_finetune = get_results(exp_name, finetuned_model, task)
breakpoint()

""" Plot the results """
results_list = [(results, "Baseline"), (results_finetune, "Finetuned")]
if linear_exists:
    results_adapted = get_results("linear", adapted_model, task)
    results_qa_adapted = get_results(f"linear_q_adapted", q_adapted_model, task)
    results_list.extend([(results_adapted, "Linear"), (results_qa_adapted, "Linear (Query-Only)")])
plot_comparison(results_list, exp_name)