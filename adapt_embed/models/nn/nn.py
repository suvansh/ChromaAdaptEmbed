import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from os.path import isdir

from typing import List, Dict
from mteb.abstasks import AbsTaskRetrieval

from adapt_embed.datasets.triplet import TripletDataset
from adapt_embed.utils import stringify_corpus_item


class NNAdapter(nn.Module):
    def __init__(self, embedding_model, embedding_size, output_size=None, hidden_sizes=[512], query_only=False, separate_embeddings=False, dummy=False):
        super().__init__()
        self.query_only = query_only
        self.separate_embeddings = separate_embeddings
        assert not (query_only and separate_embeddings), "Can't have both query_only and separate_embeddings"
        self.embed = embedding_model
        
        def gen_adapter():
            layers = []
            prev_size = embedding_size
            for size in hidden_sizes:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                prev_size = size
            layers.append(nn.Linear(prev_size, output_size or embedding_size))
            return nn.Sequential(*layers)
        if self.separate_embeddings:
            self.query_adapter = gen_adapter()
            self.doc_adapter = gen_adapter()
        else:
            self.model = gen_adapter()
        self.dummy = dummy
        if self.dummy:
            if self.separate_embeddings:
                self.query_adapter = nn.Identity()
                self.doc_adapter = nn.Identity()
            else:
                self.model = nn.Identity()

    
    def forward(self, x, is_query=True):
        if self.separate_embeddings:
            if is_query:
                return self.query_adapter(self.embed.encode(x, convert_to_tensor=True))
            else:
                return self.doc_adapter(self.embed.encode(x, convert_to_tensor=True))
        else:
            return self.model(self.embed.encode(x, convert_to_tensor=True))
    
    def encode(self, xs: List[str], batch_size: int, is_query=True, **kwargs):
        return torch.cat([self(x, is_query) for x in DataLoader(xs, batch_size=batch_size)])
    
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.encode(queries, batch_size, is_query=True, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        print("Encoding corpus...")
        if self.query_only:
            return self.embed.encode([stringify_corpus_item(item) for item in corpus], batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        else:
            return self.encode([stringify_corpus_item(item) for item in corpus], batch_size, is_query=False, **kwargs)
    
    def fit(self,
            retrieval_task_or_data: AbsTaskRetrieval | TripletDataset,
            num_epochs=10,
            log_losses=True,
            margin=1.0,
            batch_size=32,
            lr=3e-3,
            model_save_path=None,
            optimizer_class=optim.AdamW):
        
        retrieval_data = retrieval_task_or_data if isinstance(retrieval_task_or_data, TripletDataset) else TripletDataset(retrieval_task_or_data)
        train_data = DataLoader(retrieval_data, batch_size=batch_size, shuffle=True)
        loss_fn = nn.TripletMarginLoss(margin, reduction='sum')
        if not self.dummy:
            optimizer = optimizer_class(self.parameters(), lr=lr)
        
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            for query, good_doc, bad_doc in train_data:
                if not self.dummy:
                    optimizer.zero_grad()
                query_embedding = self(query, is_query=True)
                good_doc_embedding = self.embed.encode(good_doc, convert_to_tensor=True) if self.query_only else self(good_doc, is_query=False)
                bad_doc_embedding = self.embed.encode(bad_doc, convert_to_tensor=True) if self.query_only else self(bad_doc, is_query=False)
                
                loss = loss_fn(query_embedding, good_doc_embedding, bad_doc_embedding)
                if not self.dummy:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
            
            losses.append(total_loss/len(train_data))
            if log_losses:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]}")
        if model_save_path:
            if isdir(model_save_path):
                model_save_path = f"{model_save_path}/weights.pt"
            torch.save(self.state_dict(), model_save_path)
        return losses
    
    def load(self, model_save_path):
        if isdir(model_save_path):
            model_save_path = f"{model_save_path}/weights.pt"
        self.load_state_dict(torch.load(model_save_path))
