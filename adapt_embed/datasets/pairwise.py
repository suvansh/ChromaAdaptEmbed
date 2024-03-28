# TODO (suvansh): once things stabilize, figure out the right abstractions. Currently a lot of duplicate code with triplet.py
import os
import json
from datetime import datetime
import random
import torch
from torch.utils.data import Dataset
from mteb.abstasks import AbsTaskRetrieval
from mteb.evaluation.evaluators import DenseRetrievalExactSearch, DRESModel
from sentence_transformers import SentenceTransformer

from adapt_embed.utils import gen_synthetic_data, stringify_corpus_item, get_device, get_proj_dir


class PairwiseScoreDataset(Dataset):
    def __init__(self,
                 retrieval_task: AbsTaskRetrieval,
                 dres: DenseRetrievalExactSearch | None = None,
                 normalized=False,
                 thresholded=False,
                 proportional_relevance_threshold=0.5,
                 relevance_threshold=None,
                 negative_sampling=True,
                 synthetic_data=False,
                 data_augmentation_threshold=5,
                 split=None,
                 load_dir=None,
                 **load_data_kwargs):
        """
        Creates a dataset of query-document pairs and their relevance scores
        :param retrieval_task: AbsTaskRetrieval from MTEB
        :param dres: DenseRetrievalExactSearch from MTEB used to compute query-document similarity
        :param thresholded: bool representing whether to threshold the relevance scores. if True, supersedes normalized
        :param normalized: bool representing whether to normalize the relevance scores to [0, 1]
        :param proportional_relevance_threshold: float in [0, 1] representing the threshold for relevance.
            only used if negative_sampling or synthetic_data is True
        :param relevance_threshold: float representing the absolute threshold for relevance. if set, overrides proportional_relevance_threshold
        :param negative_sampling: bool representing whether to sample negative examples
        :param synthetic_data: bool representing whether to use synthetic data to generate positives. currently unused
        :param data_augmentation_threshold: int representing the number of relevant docs below which to augment data. only used if negative_sampling or synthetic_data is True
        :param split: str representing the split to use. defaults to the last eval split in the retrieval task
        :param load_dir: str representing the directory to load the dataset from
        :param load_data_kwargs: additional keyword arguments to pass to the retrieval task's load_data method
        """
        if retrieval_task is None:
            if load_dir is None:
                raise ValueError("Either retrieval_task or load_dir must be provided.")
            else:
                attributes_path = os.path.join(load_dir, 'attributes.json')
                data_path = os.path.join(load_dir, 'data.json')
                positive_doc_ids_path = os.path.join(load_dir, 'positive_doc_ids.json')
                negative_doc_ids_path = os.path.join(load_dir, 'negative_doc_ids.json')
                
                if not os.path.exists(attributes_path) or not os.path.exists(data_path):
                    raise FileNotFoundError("The specified directory does not contain the required files.")
                
                with open(attributes_path, 'r') as f:
                    attributes = json.load(f)
                    for key, value in attributes.items():
                        setattr(self, key, value)
                
                with open(data_path, 'r') as f:
                    self.data = json.load(f)
                
                with open(positive_doc_ids_path, 'r') as f:
                    self.positive_doc_ids = json.load(f)
                with open(negative_doc_ids_path, 'r') as f:
                    self.negative_doc_ids = json.load(f)
        else:
            self.retrieval_task = retrieval_task
            self.task_description = retrieval_task.description
            self.retrieval_task.load_data(**load_data_kwargs)
            
            if thresholded and normalized:
                raise ValueError("Cannot set both normalized and thresholded to True")
            self.normalized = normalized
            self.thresholded = thresholded
            
            self.proportional_relevance_threshold = proportional_relevance_threshold
            if relevance_threshold is None:
                self.set_threshold()
            else:
                self.relevance_threshold = relevance_threshold
                self.min_relevance_score = 0
                self.max_relevance_score = 1
            self.negative_sampling = negative_sampling
            self.synthetic_data = synthetic_data
            self.data_augmentation_threshold = data_augmentation_threshold

            self.split = split or self.retrieval_task.description['eval_splits'][-1]
            query_ids = self.retrieval_task.queries[self.split].keys()
            self.data = []

            """ Create a list of scored pairs for each query. """
            # if negative_sampling, augment negatives up to data_augmentation_threshold.
            # if synthetic_data, augment positives up to data_augmentation_threshold.
            negatives = {}
            positives = {}
            for query_id in query_ids:
                relevant_docs = self.retrieval_task.relevant_docs[self.split][query_id]
                negatives[query_id] = {(doc_id, score, False) for doc_id, score in relevant_docs.items() if score < self.relevance_threshold}  # False indicates doc_id
                positives[query_id] = {(doc_id, score, False) for doc_id, score in relevant_docs.items() if score >= self.relevance_threshold}  # False indicates doc_id
                if self.negative_sampling and (neg_docs_left := self.data_augmentation_threshold - len(negatives[query_id])) > 0:
                    random_docs = random.sample(list(self.retrieval_task.corpus[self.split].keys()), neg_docs_left)
                    for doc_id in random_docs:
                        negatives[query_id].add((doc_id, self.min_relevance_score, False))  # False indicates doc_id
                if self.synthetic_data and (pos_docs_left := self.data_augmentation_threshold - len(positives[query_id])) > 0:
                    synthetic_docs = gen_synthetic_data(self.retrieval_task.queries[self.split][query_id], pos_docs_left)
                    for doc in synthetic_docs:
                        positives[query_id].add((doc, self.max_relevance_score, True))  # True indicates a document itself
                for (doc_id_or_doc, score, is_doc) in positives[query_id]:
                    self.data.append((query_id, doc_id_or_doc, score, is_doc))
                for (doc_id_or_doc, score, is_doc) in negatives[query_id]:
                    self.data.append((query_id, doc_id_or_doc, score, is_doc))
            self.negative_doc_ids = negatives
            self.positive_doc_ids = positives
    
    def set_threshold(self: float):
        """
        Set the threshold for relevance.
        proportional_relevance_threshold=0.5 means that the middle of the range of relevance scores is considered relevant.
        """
        assert 0 <= self.proportional_relevance_threshold <= 1, "proportional_relevance_threshold must be in the range [0, 1]"
        max_value = float('-inf')
        min_value = float('inf')

        for relevance_splits in self.retrieval_task.relevant_docs.values():
            for d in relevance_splits.values():
                for value in d.values():
                    if value > max_value:
                        max_value = value
                    if value < min_value:
                        min_value = value
        self.relevance_threshold = min_value + (max_value - min_value) * self.proportional_relevance_threshold
        self.max_relevance_score = max_value
        self.min_relevance_score = min_value

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        query_id, doc_id_or_doc, score, is_doc = self.data[idx]
        doc_content = doc_id_or_doc if is_doc else stringify_corpus_item(self.retrieval_task.corpus[self.split][doc_id_or_doc])
        
        if self.thresholded:
            score = 1 if score >= self.relevance_threshold else 0
        elif self.normalized:
            score = (score - self.min_relevance_score) / (self.max_relevance_score - self.min_relevance_score)
        
        return (
            self.retrieval_task.queries[self.split][query_id],
            doc_content,
            score
        )

    def save(self, directory=None):
        if directory is None:
            directory = os.path.join(get_proj_dir(), 'datasets', 'cached', self.retrieval_task.description['task'], self.split, 'pairwise', datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save attributes
        attributes = {
            'proportional_relevance_threshold': self.proportional_relevance_threshold,
            'relevance_threshold': self.relevance_threshold,
            'normalized': self.normalized,
            'thresholded': self.thresholded,
            'min_relevance_score': self.min_relevance_score,
            'max_relevance_score': self.max_relevance_score,
            'negative_sampling': self.negative_sampling,
            'synthetic_data': self.synthetic_data,
            'data_augmentation_threshold': self.data_augmentation_threshold,
            'split': self.split,
            'task_description': self.task_description
        }
        with open(os.path.join(directory, 'attributes.json'), 'w') as f:
            json.dump(attributes, f, indent=4)
        
        # Save dataset
        with open(os.path.join(directory, 'data.json'), 'w') as f:
            json.dump(self.data, f, indent=None)
        
        # Save positive and negative doc ids
        with open(os.path.join(directory, 'positive_doc_ids.json'), 'w') as f:
            json.dump(self.positive_doc_ids, f, indent=None)
        with open(os.path.join(directory, 'negative_doc_ids.json'), 'w') as f:
            json.dump(self.negative_doc_ids, f, indent=None)

    @classmethod
    def load(cls, directory):
        return cls(load_dir=directory)