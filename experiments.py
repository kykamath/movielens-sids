"""
experiments.py

This module defines the configurations for different experiments using a typed dataclass.
Each experiment is an instance of the ExperimentConfig class, ensuring type safety
and providing autocompletion support.
"""
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """A typed class for defining experiment-specific configurations."""
    embedding_model_name: str
    embedding_dim: int
    hub_model_id: str

# --- Experiment Definitions ---

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        embedding_model_name='sentence-transformers/all-mpnet-base-v2',
        embedding_dim=768,
        hub_model_id="krishnakamath/rq-vae-movielens-baseline"
    ),
    "roberta-large": ExperimentConfig(
        embedding_model_name='sentence-transformers/all-roberta-large-v1',
        embedding_dim=1024,
        hub_model_id="krishnakamath/rq-vae-movielens-roberta-large"
    ),
    "kalm-gemma": ExperimentConfig(
        embedding_model_name='tencent/KaLM-Embedding-Gemma3-12B-2511',
        embedding_dim=4096,
        hub_model_id="krishnakamath/rq-vae-movielens-kalm-gemma"
    ),
    "nemotron-8b": ExperimentConfig(
        embedding_model_name='nvidia/llama-embed-nemotron-8b',
        embedding_dim=8192,
        hub_model_id="krishnakamath/rq-vae-movielens-nemotron-8b"
    )
}
