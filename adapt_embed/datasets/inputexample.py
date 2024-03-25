from torch.utils.data import Dataset
from sentence_transformers import InputExample

from adapt_embed.datasets import TripletDataset, PairwiseScoreDataset

class InputExampleDataset(Dataset):
    """
    Wraps a dataset to create a dataset of InputExamples for compatibility with SBERT cross-encoders
    """
    def __init__(self, dataset):
        self.dataset = dataset
        if isinstance(self.dataset, TripletDataset):
            self.type = "triplet"
        elif isinstance(self.dataset, PairwiseScoreDataset):
            self.type = "pairwise"
        else:
            raise ValueError("Unsupported dataset type")

    def __len__(self):
        return len(self.dataset) if self.type == "pairwise" else len(self.dataset) * 2
    
    def __getitem__(self, idx):
        if self.type == "pairwise":
            query, doc, score = self.dataset[idx]
            return InputExample(texts=[query, doc], label=float(score))
        else:
            query, good_doc, bad_doc = self.dataset.data[idx//2]
            score = 1 if idx % 2 == 0 else 0
            return InputExample(texts=[query, good_doc if idx % 2 == 0 else bad_doc], label=float(score))