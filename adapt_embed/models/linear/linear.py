import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import List, Dict
from mteb.abstasks import AbsTaskRetrieval

from triplet import TripletDataset
from utils import stringify_corpus_item


class LinearAdapter(nn.Module):
    def __init__(self, embedding_model, embedding_size, output_size=None, query_only=False):
        super().__init__()
        self.query_only = query_only
        self.embed = embedding_model
        self.linear = nn.Linear(embedding_size, output_size or embedding_size)
    
    def forward(self, x):
        return self.linear(self.embed.encode(x, convert_to_tensor=True))
    
    def encode(self, xs: List[str], batch_size: int, **kwargs):
        return torch.cat([self(x) for x in DataLoader(xs, batch_size=batch_size)])

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.encode(queries, batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        print("Encoding corpus...")
        if self.query_only:
            return self.embed.encode([stringify_corpus_item(item) for item in corpus], batch_size=batch_size, convert_to_tensor=True)
        else:
            return self.encode([stringify_corpus_item(item) for item in corpus], batch_size, **kwargs)
    
    def fit(self,
            retrieval_task_or_data: AbsTaskRetrieval | TripletDataset,
            num_epochs=10,
            log_losses=True,
            margin=1.0,
            batch_size=32,
            lr=3e-3,
            optimizer=None):
        
        retrieval_data = retrieval_task_or_data if isinstance(retrieval_task_or_data, TripletDataset) else TripletDataset(retrieval_task_or_data)
        train_data = DataLoader(retrieval_data, batch_size=batch_size, shuffle=True)
        loss_fn = nn.TripletMarginLoss(margin)
        optimizer = optim.Adam(self.linear.parameters(), lr=lr) if optimizer is None else optimizer
        
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            for query, good_doc, bad_doc in train_data:
                optimizer.zero_grad()
                query_embedding = self(query)
                good_doc_embedding = self.embed.encode(good_doc, convert_to_tensor=True) if self.query_only else self(good_doc)
                bad_doc_embedding = self.embed.encode(bad_doc, convert_to_tensor=True) if self.query_only else self(bad_doc)
                
                loss = loss_fn(query_embedding, good_doc_embedding, bad_doc_embedding)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            losses.append(total_loss/len(train_data))
            if log_losses:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]}")
        return losses
        

"""
Usage (isinstance(retrieval_task, AbsTaskRetrieval)):
# one-time overhead
eval_split = retrieval_task.description['eval_splits'][-1]
dres = DenseRetrievalExactSearch(embedding_model)
dres.search(retrieval_task.corpus[eval_split], retrieval_task.queries[eval_split], top_k=10, return_sorted=True)
dataset = TripletDataset(retrieval_task, dres=dres)
retrieval_task.evaluate(LinearAdapter(model, model_dim).fit(dataset), split=eval_split)

# each-time overhead
eval_split = retrieval_task.description['eval_splits'][-1]
retrieval_task.evaluate(LinearAdapter(model, model_dim).fit(retrieval_task), split=eval_split)
"""