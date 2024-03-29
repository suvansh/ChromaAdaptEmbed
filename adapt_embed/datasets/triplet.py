# TODO (suvansh): once things stabilize, figure out the right abstractions. Currently a lot of duplicate code with pairwise.py
from typing import Optional
import torch
import os
import json
import random
from datetime import datetime
from torch.utils.data import Dataset
from mteb.abstasks import AbsTaskRetrieval
from mteb.evaluation.evaluators import DenseRetrievalExactSearch, DRESModel
from sentence_transformers import SentenceTransformer

from adapt_embed.utils import get_proj_dir, stringify_corpus_item, get_device, gen_synthetic_data


class TripletDataset(Dataset):
    def __init__(self,
                 retrieval_task: Optional[AbsTaskRetrieval] = None,
                 dres: Optional[DenseRetrievalExactSearch] = None,
                 proportional_relevance_threshold=0.5,
                 relevance_threshold=None,
                 negative_sampling=True,
                 synthetic_data=False,
                 data_augmentation_threshold=5,
                 split=None,
                 load_dir=None,
                 **load_data_kwargs):
        """
        Creates a dataset of triplets of (query, good_document, bad_document) for training a triplet loss model.
        :param retrieval_task: AbsTaskRetrieval from MTEB.
        :param dres: DenseRetrievalExactSearch from MTEB used to compute query-document similarity
        :param proportional_relevance_threshold: float in [0, 1] representing the threshold for relevance
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
            self.task_name = retrieval_task.metadata.name
            self.retrieval_task.load_data(**load_data_kwargs)

            self.proportional_relevance_threshold = proportional_relevance_threshold
            self.set_threshold()  # run it either way to get min and max relevance scores
            if relevance_threshold is not None:
                self.relevance_threshold = relevance_threshold
            self.negative_sampling = negative_sampling
            self.synthetic_data = synthetic_data
            self.data_augmentation_threshold = data_augmentation_threshold
            
            if self.relevance_threshold is None:
                self.set_threshold()
            
            self.split = split or self.retrieval_task.description['eval_splits'][-1]
            query_ids = self.retrieval_task.queries[self.split].keys()
            self.data = []
            
            """ Create a list of triplets for each query. """
            # if negative_sampling, augment negatives up to data_augmentation_threshold.
            # if synthetic_data, augment positives up to data_augmentation_threshold.
            negatives = {}
            positives = {}
            for query_id in query_ids:
                relevant_docs = self.retrieval_task.relevant_docs[self.split][query_id]
                negatives[query_id] = {doc_id for doc_id, score in relevant_docs.items() if score < self.relevance_threshold}
                positives[query_id] = {doc_id for doc_id, score in relevant_docs.items() if score >= self.relevance_threshold}
                if self.negative_sampling and (neg_docs_left := self.data_augmentation_threshold - len(negatives[query_id])) > 0:
                    random_docs = random.sample(list(self.retrieval_task.corpus[self.split].keys()), neg_docs_left)
                    negatives[query_id].update(random_docs)
                if self.synthetic_data and (pos_docs_left := self.data_augmentation_threshold - len(positives[query_id])) > 0:
                    positives[query_id].update(gen_synthetic_data(self.retrieval_task.queries[self.split][query_id], pos_docs_left))
                for good_doc_id in positives[query_id]:
                    for bad_doc_id in negatives[query_id]:
                        self.data.append((query_id, good_doc_id, bad_doc_id))
                negatives[query_id] = list(negatives[query_id])
                positives[query_id] = list(positives[query_id])
            self.negative_doc_ids = negatives
            self.positive_doc_ids = positives


    def set_threshold(self):
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
        query_id, good_doc_id, bad_doc_id = self.data[idx]
        return (
            self.retrieval_task.queries[self.split][query_id],
            stringify_corpus_item(self.retrieval_task.corpus[self.split][good_doc_id]),
            stringify_corpus_item(self.retrieval_task.corpus[self.split][bad_doc_id])
        )

    def save(self, directory=None):
        if directory is None:
            directory = os.path.join(get_proj_dir(), 'datasets', 'cached', self.retrieval_task.description['task'], self.split, 'triplet', datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save attributes
        attributes = {
            'proportional_relevance_threshold': self.proportional_relevance_threshold,
            'relevance_threshold': self.relevance_threshold,
            'min_relevance_score': self.min_relevance_score,
            'max_relevance_score': self.max_relevance_score,
            'negative_sampling': self.negative_sampling,
            'synthetic_data': self.synthetic_data,
            'data_augmentation_threshold': self.data_augmentation_threshold,
            'split': self.split,
            'task_name': self.task_name
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
