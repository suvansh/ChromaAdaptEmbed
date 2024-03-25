def get_messages(query):
    return [
        {"role": "system", "content":
            "You will be provided with a query for an information retrieval task. "
                "You are to generate a roughly 100-word excerpt from a made-up document that could serve as a positive example "
                "of a relevant retrieved document. It should be relevant to the query and contain information that would be useful in answering the query. "
                "You are not to respond directly to the query, but rather to generate a document that could be used to answer the query. "},
        {"role": "user", "content": query}
    ]
