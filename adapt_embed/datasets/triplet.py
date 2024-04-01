from adapt_embed.datasets.base import BaseDataset
from adapt_embed.utils import stringify_corpus_item


class TripletDataset(BaseDataset):
    def load_data(self):
        super().load_data()  # populates self.positive_doc_ids and self.negative_doc_ids
        for query_id in self.retrieval_task.queries[self.split].keys():
            for good_doc_id_or_doc, _, good_is_doc in self.positive_doc_ids.get(query_id, []):
                for bad_doc_id_or_doc, _, bad_is_doc in self.negative_doc_ids.get(query_id, []):
                    self.data.append((query_id, good_doc_id_or_doc, good_is_doc, bad_doc_id_or_doc, bad_is_doc))

    def __getitem__(self, idx):
        query_id, good_doc_id_or_doc, good_is_doc, bad_doc_id_or_doc, bad_is_doc = self.data[idx]
        return (
            self.retrieval_task.queries[self.split][query_id],
            good_doc_id_or_doc if good_is_doc else stringify_corpus_item(self.retrieval_task.corpus[self.split][good_doc_id_or_doc]),
            bad_doc_id_or_doc if bad_is_doc else stringify_corpus_item(self.retrieval_task.corpus[self.split][bad_doc_id_or_doc])
        )

