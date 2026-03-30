from sentence_transformers import SentenceTransformer
import os

# Using a multilingual model to support 50+ languages as per requirements
# Using an environment variable to allow override if needed
MODEL_NAME = os.getenv('SBERT_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')

# Singleton instance
_model = None

def get_model():
    """
    Returns the SentenceTransformer model instance, loading it if necessary.
    This avoids redundant loading across different parts of the application.
    """
    global _model
    if _model is None:
        print(f"Loading SentenceTransformer model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def generate_embedding(text, model=None):
    """
    Generates a dense vector embedding for the input text.
    Returns a numpy array.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Use provided model or get the singleton
    if model is None:
        model = get_model()
    
    # Generate embedding using SBERT
    embedding = model.encode(text, show_progress_bar=False)
    return embedding

def generate_embeddings_batch(texts, model=None):
    """
    Generates dense vector embeddings for a list of texts.
    Returns a numpy array of embeddings.
    """
    if not texts or not isinstance(texts, list):
        return []
    
    # Filter out any non-string or empty elements to avoid errors
    valid_texts = [str(t) if t else "" for t in texts]
    
    # Use provided model or get the singleton
    if model is None:
        model = get_model()
    
    # Generate embeddings in batch for better performance
    embeddings = model.encode(valid_texts, batch_size=32, show_progress_bar=False)
    return embeddings
