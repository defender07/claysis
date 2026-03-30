import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(embedding_a, embedding_b):
    """
    Calculates the cosine similarity between two embeddings.
    Embeddings should be 1D or 2D numpy arrays.
    Returns a float representing the score between -1 and 1.
    """
    if len(embedding_a.shape) == 1:
        embedding_a = embedding_a.reshape(1, -1)
    if len(embedding_b.shape) == 1:
        embedding_b = embedding_b.reshape(1, -1)
        
    score = cosine_similarity(embedding_a, embedding_b)
    return float(score[0][0])

def calculate_batch_similarity(query_embedding, doc_embeddings):
    """
    Calculates cosine similarity between a query and a batch of documents.
    Returns a 1D numpy array of scores.
    """
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
        
    # docs should be a 2D array: (num_docs, embedding_dim)
    scores = cosine_similarity(query_embedding, doc_embeddings)
    return scores[0]
