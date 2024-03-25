# TODO (suvansh): once things stabilize, figure out the right abstractions. Currently a lot of duplicate code with pairwise.py
import torch
import random
from torch.utils.data import Dataset
from mteb.abstasks import AbsTaskRetrieval
from mteb.evaluation.evaluators import DenseRetrievalExactSearch, DRESModel
from sentence_transformers import SentenceTransformer

from adapt_embed.utils import stringify_corpus_item, get_device, gen_synthetic_data


class TripletDataset(Dataset):
    def __init__(self,
                 retrieval_task: AbsTaskRetrieval,
                 dres: DenseRetrievalExactSearch | None = None,
                 proportional_relevance_threshold=0.5,
                 relevance_threshold=None,
                 negative_sampling=True,
                 synthetic_data=False,
                 data_augmentation_threshold=5,
                 split=None,
                 **load_data_kwargs):
        """
        Creates a dataset of triplets of (query, good_document, bad_document) for training a triplet loss model.
        :param retrieval_task: AbsTaskRetrieval from MTEB
        :param dres: DenseRetrievalExactSearch from MTEB used to compute query-document similarity
        :param proportional_relevance_threshold: float in [0, 1] representing the threshold for relevance
        :param proportional_relevance_threshold: float in [0, 1] representing the threshold for relevance.
            only used if negative_sampling or synthetic_data is True
        :param relevance_threshold: float representing the absolute threshold for relevance. if set, overrides proportional_relevance_threshold
        :param negative_sampling: bool representing whether to sample negative examples
        :param synthetic_data: bool representing whether to use synthetic data to generate positives. currently unused
        :param data_augmentation_threshold: int representing the number of relevant docs below which to augment data
        :param split: str representing the split to use. defaults to the last eval split in the retrieval task
        :param load_data_kwargs: additional keyword arguments to pass to the retrieval task's load_data method
        """
        self.retrieval_task = retrieval_task
        self.retrieval_task.load_data(**load_data_kwargs)

        self.proportional_relevance_threshold = proportional_relevance_threshold
        self.relevance_threshold = relevance_threshold
        self.negative_sampling = negative_sampling
        self.synthetic_data = synthetic_data
        self.data_augmentation_threshold = data_augmentation_threshold
        
        if self.relevance_threshold is None:
            self.set_threshold()
        
        self.split = split or self.retrieval_task.description['eval_splits'][-1]
        query_ids = self.retrieval_task.queries[self.split].keys()
        self.data = []
        
        # if self.negative_sampling or self.synthetic_data:
        #     if dres is None:
        #         default_embed = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
        #         dres = DenseRetrievalExactSearch(DRESModel(default_embed))
        #     if not dres.results:
        #         dres.search(self.retrieval_task.corpus[self.split], self.retrieval_task.queries[self.split], top_k=100, score_function='cos_sim', return_sorted=True)
        
        # if self.negative_sampling, use dres.results to augment negative data. if self.synthetic_data then positive data
        negatives = {}
        positives = {}
        for query_id in query_ids:
            relevant_docs = self.retrieval_task.relevant_docs[self.split][query_id]
            negatives[query_id] = {doc_id for doc_id, score in relevant_docs.items() if score < self.relevance_threshold}
            positives[query_id] = {doc_id for doc_id, score in relevant_docs.items() if score >= self.relevance_threshold}
            if self.negative_sampling and (neg_docs_left := self.data_augmentation_threshold - len(negatives[query_id])) > 0:
                random_docs = random.sample(list(self.retrieval_task.corpus[self.split].keys()), neg_docs_left)
                negatives[query_id].update(random_docs)
                # for doc_id, score in dres.results[query_id].items():
                #     if score < self.relevance_threshold:
                #         negatives[query_id].add(doc_id)
                #     if len(negatives[query_id]) >= self.data_augmentation_threshold:
                #         break
            if self.synthetic_data and (pos_docs_left := self.data_augmentation_threshold - len(positives[query_id])) > 0:
                positives[query_id].update(gen_synthetic_data(self.retrieval_task.queries[self.split][query_id], pos_docs_left))
            for good_doc_id in positives[query_id]:
                for bad_doc_id in negatives[query_id]:
                    self.data.append((query_id, good_doc_id, bad_doc_id))
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