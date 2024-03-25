from adapt_embed.models.nn import NNAdapter


class LinearAdapter(NNAdapter):
    def __init__(self, embedding_model, embedding_size, output_size=None, query_only=False, **kwargs):
        super().__init__(embedding_model, embedding_size, output_size=output_size, hidden_sizes=[], query_only=query_only, **kwargs)
