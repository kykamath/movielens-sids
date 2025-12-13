import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def get_embedding_column_name(model_name: str) -> str:
    """
    Creates a sanitized and descriptive column name from a Sentence Transformer model name.
    Example: 'sentence-transformers/all-mpnet-base-v2' -> 'embedding_all_mpnet_base_v2'
    """
    # Remove the 'sentence-transformers/' prefix if it exists
    if 'sentence-transformers/' in model_name:
        model_name = model_name.split('/')[-1]
        
    # Sanitize the rest of the name
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    return f"embedding_{sanitized_name}"

def generate_embeddings(
    model_name: str,
    texts_to_embed: List[str],
    device: str = "cpu",
    show_progress_bar: bool = True
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using a specified Sentence Transformer model.

    Args:
        model_name: The name of the Sentence Transformer model from the Hugging Face Hub.
        texts_to_embed: A list of strings to be embedded.
        device: The device to run the model on ('cpu', 'cuda').
        show_progress_bar: Whether to display a progress bar during encoding.

    Returns:
        A numpy array of the generated embeddings.
    """
    print(f"Loading sentence transformer model: '{model_name}'...")
    embedding_model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts_to_embed)} texts...")
    embeddings = embedding_model.encode(
        texts_to_embed,
        show_progress_bar=show_progress_bar,
        device=device
    )
    return embeddings
