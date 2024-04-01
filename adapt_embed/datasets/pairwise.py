from typing import Optional
from mteb.abstasks import AbsTaskRetrieval

from adapt_embed.datasets.base import BaseDataset
from adapt_embed.utils import stringify_corpus_item

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PairwiseScoreDataset(BaseDataset):
    def __init__(self,
                 retrieval_task: Optional[AbsTaskRetrieval],
                 normalized=False,
                 thresholded=False,
                 **kwargs):
        """
        Creates a dataset of query-document pairs and their relevance scores
        :param retrieval_task: AbsTaskRetrieval from MTEB
        :param thresholded: bool representing whether to threshold the relevance scores. if True, supersedes normalized
        :param normalized: bool representing whether to normalize the relevance scores to [0, 1]
        """
        if thresholded and normalized:
            raise ValueError("Cannot set both normalized and thresholded to True")
        self.normalized = normalized
        self.thresholded = thresholded
        super().__init__(retrieval_task, **kwargs)
        if self.min_relevance_score == self.max_relevance_score:
            self.normalized = False
            logger.warn("`normalized` has been set False because min_relevance_score == max_relevance_score")
        
    def load_data(self):
        super().load_data()  # populates self.positive_doc_ids and self.negative_doc_ids
        for query_id in self.retrieval_task.queries[self.split].keys():
            for doc_id_or_doc, score, is_doc in self.positive_doc_ids.get(query_id, []):
                self.data.append((query_id, doc_id_or_doc, score, is_doc))
            for doc_id_or_doc, score, is_doc in self.negative_doc_ids.get(query_id, []):
                self.data.append((query_id, doc_id_or_doc, score, is_doc))

    
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
        super().save(directory=directory, thresholded=self.thresholded, normalized=self.normalized)
