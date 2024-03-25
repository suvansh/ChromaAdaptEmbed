import os
import json
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer

from adapt_embed.models.nn import NNAdapter
from adapt_embed.models.linear import LinearAdapter
from adapt_embed.utils import get_proj_dir, plot_comparison, get_device


proj_dir = get_proj_dir()
device = get_device()


""" What are we doing? """
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device=device)
exp_name = "nn"
task = 'CQADupstackEnglishRetrieval'


""" Load/train the model """
qa_weights_file = f"{proj_dir}/results/{exp_name}/{task}/qa_weights.pt"
os.makedirs(os.path.dirname(qa_weights_file), exist_ok=True)
nn_q_adapted_model = NNAdapter(model, model.get_sentence_embedding_dimension(),
                            hidden_sizes=[model.get_sentence_embedding_dimension()], query_only=True).to(device)
if os.path.exists(qa_weights_file):
    nn_q_adapted_model.load_state_dict(torch.load(qa_weights_file))
else:
    task_class = MTEB(tasks=[task]).tasks[0]
    nn_q_adapted_model.fit(task_class, num_epochs=10, lr=3e-3, batch_size=128, model_save_path=qa_weights_file)

weights_file = f"{proj_dir}/results/{exp_name}/{task}/weights.pt"
os.makedirs(os.path.dirname(weights_file), exist_ok=True)
nn_adapted_model = NNAdapter(model, model.get_sentence_embedding_dimension(),
                          hidden_sizes=[model.get_sentence_embedding_dimension()]).to(device)
if os.path.exists(weights_file):
    nn_adapted_model.load_state_dict(torch.load(weights_file))
else:
    task_class = MTEB(tasks=[task]).tasks[0]
    nn_adapted_model.fit(task_class, num_epochs=5, lr=3e-3, batch_size=128, model_save_path=weights_file)


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
        pass  # just use the NN model

""" Evaluate the model """
def get_results(exp_name, model, task):
    results_file = f"{proj_dir}/results/{exp_name}/{task}/{task}.json"
    if os.path.exists(results_file):
        with open(results_file) as json_file:
            return {task: json.load(json_file)}
    else:
        return MTEB(tasks=[task]).run(model, output_folder=os.path.dirname(results_file))

results = get_results("embed_baseline", model, task)
results_adapted = get_results(exp_name, nn_adapted_model, task)
results_qa_adapted = get_results(f"{exp_name}_q_adapted", nn_q_adapted_model, task)

""" Plot the results """
results_list = [(results, "Baseline"), (results_adapted, "NN"), (results_qa_adapted, "NN (Query-Only)")]
if linear_exists:
    results_linear = get_results("linear", adapted_model, task)
    results_qa_linear = get_results("linear_q_adapted", q_adapted_model, task)
    results_list.extend([(results_linear, "Linear"), (results_qa_linear, "Linear (Query-Only)")])
plot_comparison(results_list, exp_name)