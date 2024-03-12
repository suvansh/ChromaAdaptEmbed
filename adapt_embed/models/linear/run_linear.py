import os
import json
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer

from linear import LinearAdapter
from utils import get_proj_dir, plot_comparison

proj_dir = get_proj_dir()

""" What are we doing? """
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
exp_name = "linear"
task = 'CQADupstackEnglishRetrieval'


""" Load/train the model """
weights_file = f"{proj_dir}/results/{exp_name}/{task}/weights.pt"
adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension())
if os.path.exists(weights_file):
    adapted_model.load_state_dict(torch.load(weights_file))
else:
    task_class = MTEB(tasks=[task]).tasks[0]
    adapted_model.fit(task_class, num_epochs=10)
    torch.save(adapted_model.state_dict(), weights_file)

qa_weights_file = f"{proj_dir}/results/{exp_name}/{task}/qa_weights.pt"
q_adapted_model = LinearAdapter(model, model.get_sentence_embedding_dimension(), query_only=True)
if os.path.exists(qa_weights_file):
    q_adapted_model.load_state_dict(torch.load(qa_weights_file))
else:
    task_class = MTEB(tasks=[task]).tasks[0]
    q_adapted_model.fit(task_class, num_epochs=10)
    torch.save(q_adapted_model.state_dict(), qa_weights_file)

""" Evaluate the model """
def get_results(exp_name, model, task):
    results_file = f"{proj_dir}/results/{exp_name}/{task}/{task}.json"
    if os.path.exists(results_file):
        with open(results_file) as json_file:
            return {task: json.load(json_file)}
    else:
        return MTEB(tasks=[task]).run(model, output_folder=os.path.dirname(results_file))

results = get_results("embed_baseline", model, task)
results_adapted = get_results(exp_name, adapted_model, task)
results_qa_adapted = get_results(f"{exp_name}_q_adapted", q_adapted_model, task)

""" Plot the results """
plot_comparison([(results, "Baseline"), (results_adapted, "Linear"), (results_qa_adapted, "Linear (Query-Only)")], exp_name)