import os
import json
import logging
from tqdm import tqdm
from time import time
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.evaluation.evaluators import RetrievalEvaluator
from mteb.abstasks.TaskMetadata import TaskMetadata

from adapt_embed.models.reranker import Reranker


logger = logging.getLogger(__name__)

class RerankerRetrievalTask(AbsTaskRetrieval):
    metadata = TaskMetadata(        
        name="CQADupstackEnglishRetrieval",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        hf_hub_name="mteb/cqadupstack-english",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        revision="ad9991cb51e31e31e430383c75ffb2885547b5f0",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
    
    def evaluate(
        self,
        model,
        split="test",
        **kwargs
    ):
        if isinstance(model, Reranker):
            return self._evaluate_monolingual_reranker(model, self.corpus[split], self.queries[split], self.relevant_docs[split], **kwargs)
        else:
            return super().evaluate(model, split, **kwargs)

    def _evaluate_monolingual_reranker(self, model, corpus, queries, relevant_docs, top_k=100, batch_size=32, **kwargs):
        retriever = RetrievalEvaluator(model, **kwargs)
        
        scores = {}
        start_time = time()
        results = {}
        for query_id, query_text in tqdm(list(queries.items())[:1]):
            query_corpus = [(query_text, corpus[doc_id]['text']) for doc_id in corpus.keys()]
            query_scores = model.model.predict(query_corpus, show_progress_bar=True, batch_size=batch_size)
            top_docs = sorted(zip(corpus.keys(), query_scores), key=lambda x: x[1], reverse=True)[:top_k]
            results[query_id] = {doc_id: float(score) for doc_id, score in top_docs}
        end_time = time()
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        if kwargs.get('save_qrels', False):
            output_folder = kwargs.get('output_folder', 'results')
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            qrels_save_path = f"{output_folder}/{self.description['name']}_qrels.json"
            with open(qrels_save_path, 'w') as f:
                json.dump(results, f)
        
        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores
    
    @staticmethod
    def from_task(retrieval_task_cls):
        if not issubclass(retrieval_task_cls, AbsTaskRetrieval):
            raise ValueError(f"{retrieval_task_cls} is not a subclass of AbsTaskRetrieval")

        class RerankerRetrievalTaskWrapper(RerankerRetrievalTask, retrieval_task_cls):
            @property
            def description(self):
                return retrieval_task_cls.description.fget(self)

            def load_data(self, **kwargs):
                return retrieval_task_cls.load_data(self, **kwargs)

        RerankerRetrievalTaskWrapper.__name__ = f"Reranker{retrieval_task_cls.__name__}"
        return RerankerRetrievalTaskWrapper