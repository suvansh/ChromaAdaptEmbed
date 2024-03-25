import os
import json
import torch
import numpy as np
from mteb import MTEB
from sentence_transformers import SentenceTransformer

from adapt_embed.eval.reranker_retrieval import RerankerRetrievalTask
from adapt_embed.models.linear.linear import LinearAdapter
from adapt_embed.models.reranker.reranker import Reranker
from adapt_embed.datasets import PairwiseScoreDataset
from adapt_embed.datasets.inputexample import InputExampleDataset
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device

proj_dir = get_proj_dir()
device = get_device()

""" What are we doing? """
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device=device)
rerank_model_name = "distilroberta-base"
exp_name = "reranker"
# task = 'BSARDRetrieval'
task = 'CQADupstackEnglishRetrieval'
split = 'test'


""" Load/train the model """
# baseline
reranker_baseline_model = Reranker(rerank_model_name).to(device)  # this one won't be trained

reranker_weights_dir = f"{proj_dir}/results/{exp_name}/{task}/model"
reranker_model = Reranker(rerank_model_name).to(device)
# reranker_model = Reranker(reranker_weights_dir).to(device)
task_class = MTEB(tasks=[task]).tasks[0]
reranker_task_class = RerankerRetrievalTask.from_task(type(task_class))()
reranker_task_class.load_data()
dataset = InputExampleDataset(PairwiseScoreDataset(task_class, thresholded=True, relevance_threshold=0.7, negative_sampling=True, split=split))
reranker_model.fit(dataset, num_epochs=5, lr=3e-3, batch_size=128, model_save_path=reranker_weights_dir)


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
        print("Loaded linear models.")
    except:
        print("Failed to load linear models. Proceeding with reranker models.")

""" Evaluate the model """
def get_results(exp_name, model, task, **kwargs):
    results_file = f"{proj_dir}/results/{exp_name}/{task}/{task}.json"
    if os.path.exists(results_file):
        with open(results_file) as json_file:
            return {task: json.load(json_file)}
    else:
        return MTEB(tasks=[reranker_task_class]).run(model, output_folder=os.path.dirname(results_file), **kwargs)

results = get_results("embed_baseline", model, task)
def get_reranker_predictor(reranker: Reranker):
    """ queries and documents are numpy arrays of strings. return a function that computes the score on each pair """ 
    return lambda qs, ds: np.stack([reranker.model.predict([q]*len(ds), ds) for q in qs], axis=0)  # (n_q, n_d)
results_reranked = get_results(exp_name, reranker_model, task, save_qrels=True)
results_reranked_baseline = get_results(f"{exp_name}_baseline", reranker_baseline_model, task, save_qrels=True)

""" Plot the results """
results_list = [(results, "Embedding-Only Baseline"), (results_reranked_baseline, "Pre-Trained Reranker"), (results_reranked, "Fine-tuned Reranker")]
if linear_exists:
    results_linear = get_results("linear", adapted_model, task)
    results_qa_linear = get_results("linear_q_adapted", q_adapted_model, task)
    results_list.extend([(results_linear, "Linear Adapter"), (results_qa_linear, "Linear Adapter (Query-Only)")])
plot_comparison(results_list, exp_name)