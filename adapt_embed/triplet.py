import torch
from mteb.abstasks import AbsTaskRetrieval
from mteb.evaluation.evaluators import DenseRetrievalExactSearch, DRESModel
from sentence_transformers import SentenceTransformer

from utils import stringify_corpus_item

class TripletDataset:
    def __init__(self,
                 retrieval_task: AbsTaskRetrieval,
                 dres: DenseRetrievalExactSearch | None = None,
                 proportional_relevance_threshold=0.5,
                 synthetic_data=False,
                 split=None,
                 **load_data_kwargs):
        self.retrieval_task = retrieval_task
        self.retrieval_task.load_data(**load_data_kwargs)

        self.proportional_relevance_threshold = proportional_relevance_threshold
        self.synthetic_data = synthetic_data

        self.sep = "\n"
        
        self.set_threshold(proportional_relevance_threshold)
        if split is None:
            split = self.retrieval_task.description['eval_splits'][-1]
        self.split = split
        query_ids = self.retrieval_task.queries[self.split].keys()
        self.data = []
        
        if dres is None:
            default_embed = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('mps'))
            dres = DenseRetrievalExactSearch(DRESModel(default_embed))
        if not dres.results:
            # NOTE (suvansh): I'm not convinced return_sorted actually does anything since you're ultimately getting back a dict (?)
            dres.search(self.retrieval_task.corpus[self.split], self.retrieval_task.queries[self.split], top_k=50, score_function='cos_sim', return_sorted=True)
        
        # use dres.results to augment at least negative data (and if self.synthetic_data then also positive data)
        negatives = {}
        positives = {}
        for query_id in query_ids:
            relevant_docs = self.retrieval_task.relevant_docs[self.split][query_id]
            # if there are fewer than 5 relevant docs below the threshold, add docs below the threshold to the extra_negatives until there are 5
            negatives[query_id] = {doc_id for doc_id, score in relevant_docs.items() if score < self.relevance_threshold}
            positives[query_id] = {doc_id for doc_id, score in relevant_docs.items() if score >= self.relevance_threshold}
            if len(negatives[query_id]) < 5:
                for doc_id, score in dres.results[query_id].items():
                    if score < self.relevance_threshold:
                        negatives[query_id].add(doc_id)
                    if len(negatives[query_id]) >= 5:
                        break
            if self.synthetic_data and len(positives[query_id]) < 5:
                pass
            for good_doc_id in positives[query_id]:
                for bad_doc_id in negatives[query_id]:
                    self.data.append((query_id, good_doc_id, bad_doc_id))
        self.negative_doc_ids = negatives
        self.positive_doc_ids = positives


    def set_threshold(self, proportional_relevance_threshold: float):
        """
        Set the threshold for relevance.
        proportional_relevance_threshold=0.5 means that the middle of the range of relevance scores is considered relevant.
        """
        assert 0 <= proportional_relevance_threshold <= 1, "proportional_relevance_threshold must be in the range [0, 1]"
        max_value = float('-inf')
        min_value = float('inf')

        for relevance_splits in self.retrieval_task.relevant_docs.values():
            for d in relevance_splits.values():
                for value in d.values():
                    if value > max_value:
                        max_value = value
                    if value < min_value:
                        min_value = value
        self.relevance_threshold = min_value + (max_value - min_value) * proportional_relevance_threshold
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_id, good_doc_id, bad_doc_id = self.data[idx]
        return (
            self.retrieval_task.queries[self.split][query_id],
            stringify_corpus_item(self.retrieval_task.corpus[self.split][good_doc_id]),
            stringify_corpus_item(self.retrieval_task.corpus[self.split][bad_doc_id])
        )