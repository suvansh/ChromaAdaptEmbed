from abc import ABC, abstractmethod
import os
import json
from datetime import datetime
import random
from typing import Optional
import torch
from torch.utils.data import Dataset
from mteb.abstasks import AbsTaskRetrieval
from mteb.evaluation.evaluators import DenseRetrievalExactSearch, DRESModel
from sentence_transformers import SentenceTransformer

from adapt_embed.utils import gen_synthetic_data, stringify_corpus_item, get_device, get_proj_dir

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 retrieval_task: Optional[AbsTaskRetrieval],
                 proportional_relevance_threshold=0.5,
                 relevance_threshold=None,
                 negative_sampling=True,
                 synthetic_data=False,
                 synthetic_data_path=None,
                 llm="gpt-4-turbo-preview",
                 data_augmentation_threshold=5,
                 split=None,
                 load_dir=None,
                 **load_data_kwargs):
        """
        Creates a dataset of query-document pairs and their relevance scores
        :param retrieval_task: AbsTaskRetrieval from MTEB
        :param proportional_relevance_threshold: float in [0, 1] representing the threshold for relevance.
            only used if negative_sampling or synthetic_data is True
        :param relevance_threshold: float representing the absolute threshold for relevance. if set, overrides proportional_relevance_threshold
        :param negative_sampling: bool representing whether to sample negative examples
        :param synthetic_data: bool representing whether to use synthetic data to generate positives. currently unused
        :param synthetic_data_path: str representing the path to a json file containing synthetic data for each query.
            if synthetic_data is True:
                if synthetic_data_path is a valid string path to a json file, it is read as a dictionary mapping query ids to data
                if synthetic_data_path is None, synthetic data is generated with LLM and saved to a default path.
                if synthetic_data_path is a string path but doesn't exist, synthetic data is generated with LLM and saved to the path
            if synthetic_data is False: this parameter has no effect
        :param llm: str representing the LLM to use for generating synthetic data. only used if synthetic data is generated
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
            self.split = split or self.retrieval_task.description['eval_splits'][-1]
            
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
            self.llm = llm
            self.synthetic_data_path = synthetic_data_path or os.path.join(get_proj_dir(), "data", self.task_name, f"synthetic_data_{self.llm}.json")
            if synthetic_data and not os.path.isfile(self.synthetic_data_path):
                logger.warn(f"Generating up to {self.data_augmentation_threshold} synthetic documents each for {len(self.retrieval_task.queries[self.split])} queries with {self.llm}. This may take a while and cost money.")

            self.data = []
            self.negative_doc_ids = {}
            self.positive_doc_ids = {}          
            self.load_data()

    def load_data(self):
        query_ids = self.retrieval_task.queries[self.split].keys()
        self.data = []

        if os.path.isfile(self.synthetic_data_path):
            have_external_synthetic_data = True
            with open(self.synthetic_data_path, 'r') as f:
                synthetic_data_external = json.load(f)
            logger.info(f"Loaded synthetic data for task {self.task_name} from {self.synthetic_data_path}")
        else:
            have_external_synthetic_data = False
            # create the directory to save generated synthetic data to
            os.makedirs(os.path.dirname(self.synthetic_data_path), exist_ok=True)
            synthetic_data_external = {}

        # if negative_sampling, augment negatives up to data_augmentation_threshold.
        # if synthetic_data, augment positives up to data_augmentation_threshold.
        negatives = {}
        positives = {}
        examples, num_examples = [], 10
        for query_id in query_ids:
            relevant_docs = self.retrieval_task.relevant_docs[self.split][query_id]
            negatives[query_id] = {(doc_id, score, False) for doc_id, score in relevant_docs.items() if score < self.relevance_threshold}  # False indicates doc_id
            positives[query_id] = {(doc_id, score, False) for doc_id, score in relevant_docs.items() if score >= self.relevance_threshold}  # False indicates doc_id
            if len(examples) < num_examples:
                # no more than 2 examples from a given query to diversify the examples
                examples.extend([(self.retrieval_task.queries[self.split][query_id], stringify_corpus_item(self.retrieval_task.corpus[self.split][doc_id])) for doc_id, *_ in positives[query_id]][:2])
                # truncate 
                examples = examples[:num_examples]
            if self.negative_sampling and (neg_docs_left := self.data_augmentation_threshold - len(negatives[query_id])) > 0:
                random_docs = random.sample(list(self.retrieval_task.corpus[self.split].keys()), neg_docs_left)
                for doc_id in random_docs:
                    negatives[query_id].add((doc_id, self.min_relevance_score, False))  # False indicates doc_id
            if self.synthetic_data and (pos_docs_left := self.data_augmentation_threshold - len(positives[query_id])) > 0:
                if have_external_synthetic_data:
                    for doc in synthetic_data_external.get(query_id, []):
                        positives[query_id].add((doc, self.max_relevance_score, True))  # True indicates a document itself
                else:  # need to generate synthetic data
                    synthetic_docs = gen_synthetic_data(self.retrieval_task.queries[self.split][query_id],
                                                        pos_docs_left,
                                                        examples=examples if len(examples) == num_examples else None,
                                                        llm=self.llm)
                    for doc in synthetic_docs:
                        positives[query_id].add((doc, self.max_relevance_score, True))  # True indicates a document itself
                    synthetic_data_external[query_id] = synthetic_docs  # to be saved
            negatives[query_id] = list(negatives[query_id])
            positives[query_id] = list(positives[query_id])
        self.negative_doc_ids = negatives
        self.positive_doc_ids = positives
        if not have_external_synthetic_data:
            # save the generated synthetic_data_external
            with open(self.synthetic_data_path, 'w') as f:
                json.dump(synthetic_data_external, f)
            logger.info(f"Saved synthetic data for task {self.task_name} to {self.synthetic_data_path}")
    
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
        raise NotImplementedError

    def save(self, directory=None, **kwargs):
        if directory is None:
            directory = os.path.join(get_proj_dir(), 'datasets', 'cached', self.retrieval_task.description['task'], self.split, self.__class__.__name__, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save attributes
        attributes = {
            'proportional_relevance_threshold': self.proportional_relevance_threshold,
            'relevance_threshold': self.relevance_threshold,
            # NOTE
            # 'normalized': self.normalized,
            # 'thresholded': self.thresholded,
            'min_relevance_score': self.min_relevance_score,
            'max_relevance_score': self.max_relevance_score,
            'negative_sampling': self.negative_sampling,
            'synthetic_data': self.synthetic_data,
            'synthetic_data_path': self.synthetic_data_path,
            'llm': self.llm,
            'data_augmentation_threshold': self.data_augmentation_threshold,
            'split': self.split,
            'task_name': self.task_name,
            **kwargs
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