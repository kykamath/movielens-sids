from dataclasses import dataclass, field
from typing import List

# --- Hugging Face Hub Repository IDs ---
HUB_ENRICHED_REPO_ID = "krishnakamath/movielens-32m-movies-enriched"
HUB_EMBEDDINGS_REPO_ID = "krishnakamath/movielens-32m-movies-enriched-embeddings"
HUB_SIDS_DATASET_ID = "krishnakamath/movielens-32m-movies-enriched-with-SIDs"
HUB_MODEL_ID = "krishnakamath/rq-vae-movielens"

# --- Model Configuration ---
# You can easily switch between models by changing which line is commented out.
EMBEDDING_MODEL_NAME_BASE = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_MODEL_NAME_LARGE = 'sentence-transformers/all-roberta-large-v1'
EMBEDDING_MODEL_NAME_KALM = 'tencent/KaLM-Embedding-Gemma3-12B-2511' # A powerful, large-scale embedding model

# The model currently being used for generation.
# Change this to experiment with different embedding models.
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME_BASE

@dataclass
class Movie:
    movie_id: int
    title: str
    genres: List[str]
    plot_summary: str = ""
    director: str = ""
    stars: List[str] = field(default_factory=list)

    def to_embedding_string(self) -> str:
        """
        Creates a string representation of the movie for generating embeddings.
        The string is formatted to help the sentence transformer understand the semantic fields.
        """
        genres_str = ", ".join(self.genres)
        stars_str = ", ".join(self.stars)
        
        embedding_text = (
            f"Movie Title: {self.title}. "
            f"Genres: {genres_str}. "
            f"Directed by: {self.director}. "
            f"Starring: {stars_str}. "
            f"Plot Summary: {self.plot_summary}"
        )
        return embedding_text
