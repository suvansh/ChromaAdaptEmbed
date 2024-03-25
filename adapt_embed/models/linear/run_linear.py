# NOTE: deprecated in favor of run_linear_launcher.py
import os
import json
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer

from adapt_embed.datasets.triplet import TripletDataset
from adapt_embed.models.linear.linear import LinearAdapter
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device

proj_dir = get_proj_dir()
device = get_device()

""" What are we doing? """
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device=device)
exp_name = "linear"
task = 'CQADupstackEnglishRetrieval'
# task = 'BSARDRetrieval'
split = 'test'

dataset = None
def get_dataset():
    global dataset
    if dataset is None:
        dataset = TripletDataset(MTEB(tasks=[task]).tasks[0], negative_sampling=True, split=split, relevance_threshold=0.5)
    return dataset

""" Load/train the model """
qa_weights_file = f"{proj_dir}/results/{exp_name}/{task}/qa_weights.pt"
os.makedirs(os.path.dirname(qa_weights_file), exist_ok=True)
q_adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension(), query_only=True).to(device)
if os.path.exists(qa_weights_file):
    q_adapted_model.load_state_dict(torch.load(qa_weights_file))
else:
    print("Training Query-Only Linear Adapter...")
    q_adapted_model.fit(get_dataset(), num_epochs=10, lr=3e-3, batch_size=128, model_save_path=qa_weights_file)

weights_file = f"{proj_dir}/results/{exp_name}/{task}/weights.pt"
os.makedirs(os.path.dirname(weights_file), exist_ok=True)
adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension()).to(device)
if os.path.exists(weights_file):
    adapted_model.load_state_dict(torch.load(weights_file))
else:
    print("Training Linear Adapter...")
    adapted_model.fit(get_dataset(), num_epochs=10, lr=3e-3, batch_size=128, model_save_path=weights_file)

query_first_weights_file = f"{proj_dir}/results/{exp_name}/{task}/query_first_weights.pt"
os.makedirs(os.path.dirname(query_first_weights_file), exist_ok=True)
query_first_adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension()).to(device)
if os.path.exists(query_first_weights_file):
    query_first_adapted_model.load_state_dict(torch.load(query_first_weights_file))
else:
    # load qa_weights
    qa_weights = torch.load(qa_weights_file)
    with torch.no_grad():
        query_first_adapted_model.model[0].weight.copy_(qa_weights['model.0.weight'])
        query_first_adapted_model.model[0].bias.copy_(qa_weights['model.0.bias'])
    print("Training Query-First Linear Adapter...")
    query_first_adapted_model.fit(get_dataset(), num_epochs=5, lr=3e-3, batch_size=128, model_save_path=query_first_weights_file)

separate_weights_file = f"{proj_dir}/results/{exp_name}/{task}/separate_weights.pt"
os.makedirs(os.path.dirname(separate_weights_file), exist_ok=True)
separate_adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension(), separate_embeddings=True).to(device)
if os.path.exists(separate_weights_file):
    separate_adapted_model.load_state_dict(torch.load(separate_weights_file))
else:
    print("Training Separate Linear Adapter...")
    separate_adapted_model.fit(get_dataset(), num_epochs=10, lr=3e-3, batch_size=128, model_save_path=separate_weights_file)

""" Evaluate the model """
def get_results(exp_name, model, task):
    results_file = f"{proj_dir}/results/{exp_name}/{task}/{task}.json"
    # if os.path.exists(results_file):
    #     with open(results_file) as json_file:
    #         return {task: json.load(json_file)}
    # else:
    return MTEB(tasks=[task]).run(model, output_folder=os.path.dirname(results_file))

results = get_results("embed_baseline", model, task)
results_adapted = get_results(exp_name, adapted_model, task)
results_qa_adapted = get_results(f"{exp_name}_q_adapted", q_adapted_model, task)
results_query_first_adapted = get_results(f"{exp_name}_query_first", query_first_adapted_model, task)
results_separate_adapted = get_results(f"{exp_name}_separate", separate_adapted_model, task)

""" Plot the results """
plot_comparison([(results, "Baseline"), (results_adapted, "Linear"), (results_qa_adapted, "Linear (Query-Only)"), (results_query_first_adapted, "Linear (Query-First)"), (results_separate_adapted, "Linear (Separate Query/Doc)")], exp_name)