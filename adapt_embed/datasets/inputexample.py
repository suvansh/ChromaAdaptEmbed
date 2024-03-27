from torch.utils.data import Dataset
from sentence_transformers import InputExample

from adapt_embed.datasets import TripletDataset, PairwiseScoreDataset

class InputExampleDataset(Dataset):
    """
    Wraps a dataset to create a dataset of InputExamples for compatibility with SBERT cross-encoders
    """
    def __init__(self, dataset, score_triplet=False):
        self.dataset = dataset
        if isinstance(self.dataset, TripletDataset):
            if score_triplet:
                self.type = "triplet_score"
            else:
                self.type = "triplet"
        elif isinstance(self.dataset, PairwiseScoreDataset):
            self.type = "pairwise"
        else:
            raise ValueError("Unsupported dataset type")

    def __len__(self):
        return len(self.dataset) * 2 if self.type == "triplet_score" else len(self.dataset)
    
    def __getitem__(self, idx):
        if self.type == "pairwise":
            query, doc, score = self.dataset[idx]
            return InputExample(texts=[query, doc], label=float(score))
        elif self.type == "triplet_score":
            query, good_doc, bad_doc = self.dataset[idx//2]
            score = 1 if idx % 2 == 0 else 0
            return InputExample(texts=[query, good_doc if idx % 2 == 0 else bad_doc], label=float(score))
        elif self.type == "triplet":
            query, good_doc, bad_doc = self.dataset[idx]
            return InputExample(texts=[query, good_doc, bad_doc])
        else:
            raise ValueError(f"Unsupported dataset type: {self.type}")