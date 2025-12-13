"""
experiments.py

This module defines the configurations for different experiments using a typed dataclass.
Each experiment is an instance of the ExperimentConfig class, ensuring type safety
and providing autocompletion support. All models will be pushed to a single
repository and versioned using tags based on the experiment name.
"""
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """A typed class for defining experiment-specific configurations."""
    embedding_model_name: str
    embedding_dim: int

# --- Experiment Definitions ---

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "mpnet-base": ExperimentConfig(
        embedding_model_name='sentence-transformers/all-mpnet-base-v2',
        embedding_dim=768
    ),
    "roberta-large": ExperimentConfig(
        embedding_model_name='sentence-transformers/all-roberta-large-v1',
        embedding_dim=1024
    ),
    "kalm-gemma": ExperimentConfig(
        embedding_model_name='tencent/KaLM-Embedding-Gemma3-12B-2511',
        embedding_dim=4096
    ),
    "nemotron-8b": ExperimentConfig(
        embedding_model_name='nvidia/llama-embed-nemotron-8b',
        embedding_dim=8192
    )
}
