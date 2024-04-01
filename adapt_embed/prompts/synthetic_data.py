def get_messages(query, examples=None):
    """
    :param examples: list of 2-tuples of (query, positive) or None
    """
    if not examples:
        return [
            {"role": "system", "content":
                "You will be provided with a query for an information retrieval task. "
                    "You are to generate a roughly 50-word excerpt from a made-up document that could serve as a positive example "
                    "of a relevant retrieved document. It should be relevant to the query and contain information that would be useful in answering the query. "
                    "You are not to respond directly to the query, but rather to generate a document that could be used to answer the query. "
                    "Do not say anything other than the excerpt, and do not place it in quotes. Simply output the document directly."},
            {"role": "user", "content": query}
        ]
    else:
        examples_str = "\n\n".join([f"Query: {q}\nPositive: {p}" for q, p in examples])
    
        return [
            {"role": "system", "content":
                f"Below are {len(examples)} example queries followed by excerpts from made-up documents that are considered relevant and useful for an information retrieval task:\n\n{examples_str}\n\n"
                    "After reviewing these examples, you will be provided with a new query. Your task is to generate an excerpt of similar length to the examples from a fictional document that could serve as a positive example "
                    "of a relevant retrieved document for the given query. The generated document should be relevant to the query and contain information that would be useful in answering the query. "
                    "Aim to ensure that your generation is in-distribution with the examples provided, without anchoring too strongly on the specifics of these examples. "
                    "Remember, you are not to respond directly to the query, but rather to generate a document that could be used to answer the query. "
                    "Do not say anything other than the excerpt, and do not place it in quotes. Simply output the document directly."},
            {"role": "user", "content": f"Query: {query}\nPositive: "}
        ]