import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from os.path import isfile, dirname

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from mteb.abstasks import AbsTaskRetrieval

from adapt_embed.datasets import PairwiseScoreDataset, TripletDataset
from adapt_embed.datasets.inputexample import InputExampleDataset
from adapt_embed.utils import stringify_corpus_item


class Reranker:
    def __init__(self, ce_model_or_save_path: str):
        super().__init__()
        self.model = CrossEncoder(ce_model_or_save_path, num_labels=1)

    def encode(self, xs: list[str], batch_size: int, **kwargs):
        # identity because model runs on string pairs
        return xs
    
    def to(self, device):
        self.model._target_device = torch.device(device)
        return self
    
    def fit(self,
            retrieval_task_or_data: InputExampleDataset | TripletDataset | PairwiseScoreDataset | AbsTaskRetrieval,  # TODO (suvansh) make this dataset class
            num_epochs=10,
            log_losses=True,
            loss: str="bce",
            margin=1.0,
            batch_size=32,
            lr=3e-3,
            model_save_path=None,
            optimizer_class=optim.AdamW,
            **kwargs):
        
        assert loss in ["bce", "triplet"], "Loss must be either 'bce' or 'triplet'"
        if loss == "bce":
            loss_fn = nn.BCELoss()
            if isinstance(retrieval_task_or_data, InputExampleDataset):
                retrieval_data = retrieval_task_or_data
            elif isinstance(retrieval_task_or_data, PairwiseScoreDataset):
                retrieval_data = InputExampleDataset(retrieval_task_or_data)
            elif isinstance(retrieval_task_or_data, TripletDataset):
                raise ValueError("TripletDataset not supported for BCE loss. Use 'triplet' loss or convert to PairwiseScoreDataset.")
            else:
                retrieval_data = InputExampleDataset(PairwiseScoreDataset(retrieval_task_or_data))
        else:
            loss_fn = nn.TripletMarginLoss(margin, reduction='sum')
            if isinstance(retrieval_task_or_data, InputExampleDataset):
                retrieval_data = retrieval_task_or_data
            elif isinstance(retrieval_task_or_data, PairwiseScoreDataset):
                raise ValueError("PairwiseScoreDataset not supported for triplet loss. Use 'bce' loss or convert to TripletDataset.")
            elif isinstance(retrieval_task_or_data, TripletDataset):
                retrieval_data = InputExampleDataset(retrieval_task_or_data)
            else:
                retrieval_data = InputExampleDataset(TripletDataset(retrieval_task_or_data))
        train_data = DataLoader(retrieval_data, batch_size=batch_size, shuffle=True)
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(retrieval_data)
        warmup_steps = int(len(train_data) * num_epochs * 0.1)  # 10% of train data for warm-up
        self.model.fit(
            train_dataloader=train_data,
            evaluator=evaluator,
            epochs=num_epochs,
            optimizer_params={ "lr": lr },
            optimizer_class=optimizer_class,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            **kwargs
        )
        # TODO (suvansh) get it to use the diff losses

    def load(self, model_save_path):
        if isfile(model_save_path):
            model_save_path = dirname(model_save_path)
        self.model = CrossEncoder(model_save_path, num_labels=1)
