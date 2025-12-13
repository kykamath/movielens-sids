"""
evaluate.py

This script provides tools to evaluate and compare the semantic quality of models
from different experiments.

Currently, it supports a semantic similarity search, which for a given movie,
finds the top-N closest movies based on the Hamming distance of their SIDs.
This allows for a qualitative comparison of how well different models capture
the "meaning" of a movie.

This script can be run from the command line with the following arguments:

--experiments [name1] [name2] ...
    Description:
        A required list of one or more experiment names to evaluate and compare.
        The script will load the corresponding model version for each experiment
        from the Hugging Face Hub (based on the version tag).

    Usage:
        # Compare the baseline and roberta-large models
        python evaluate.py --experiments baseline roberta-large

        # Evaluate a single experiment
        python evaluate.py --experiments nemotron-8b
"""

import argparse
import torch
import numpy as np
from scipy.spatial.distance import cdist

from pipeline import PipelineConfig, DatasetManager, EmbeddingManager, RQVAEOrchestrator
from experiments import EXPERIMENTS

# --- Configuration ---
# A few interesting movies to test the semantic search on.
PROBE_MOVIES = ["The Matrix", "Toy Story", "The Godfather", "Pulp Fiction"]
TOP_K = 5 # Number of nearest neighbors to find

def find_nearest_neighbors(sids: np.ndarray, query_sid: np.ndarray, k: int):
    """Finds the top-k nearest neighbors using Hamming distance."""
    # cdist with 'hamming' calculates the proportion of differing bits
    distances = cdist(query_sid.reshape(1, -1), sids, metric='hamming')[0]
    # Get the indices of the top-k smallest distances
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices, distances[nearest_indices]

def main():
    parser = argparse.ArgumentParser(
        description="Compare semantic quality of models from different experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--experiments', 
        type=str, 
        nargs='+', # Accepts one or more experiment names
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help='A list of experiment names to compare.'
    )
    args = parser.parse_args()

    # --- 1. Load Data ---
    # We only need to do this once, as all models use the same source data.
    print("--- Loading source data... ---")
    base_config = PipelineConfig() # Default config is fine for loading data
    dataset_manager = DatasetManager(base_config)
    source_dataset = dataset_manager.load_source_dataset()
    
    # Create a quick lookup from title to item
    title_to_item = {item['title']: item for item in source_dataset}
    movie_titles = [item['title'] for item in source_dataset]

    # --- 2. Iterate Through Experiments and Evaluate ---
    for experiment_name in args.experiments:
        print(f"\n\n--- Evaluating Experiment: {experiment_name} ---")
        
        # A. Configure and load the specific model for the experiment
        experiment_config = EXPERIMENTS[experiment_name]
        config = PipelineConfig(experiment_name=experiment_name, experiment_config=experiment_config)
        model_orchestrator = RQVAEOrchestrator(config)

        # B. We need to generate embeddings using the *correct* model for this experiment
        print(f"Generating embeddings with this experiment's model: {config.embedding_model_name}")
        exp_embedding_manager = EmbeddingManager(config)
        embeddings_tensor, _, _ = exp_embedding_manager.generate(source_dataset)

        print("Loading pre-trained model and generating SIDs...")
        rq_model = model_orchestrator.load_from_hub()
        sids = np.array(model_orchestrator.generate_sids(rq_model, embeddings_tensor))

        # C. Run the similarity search for each probe movie
        for movie_title in PROBE_MOVIES:
            if movie_title not in title_to_item:
                print(f"\nWarning: Probe movie '{movie_title}' not found in dataset.")
                continue

            print(f"\nResults for '{movie_title}':")
            
            query_idx = movie_titles.index(movie_title)
            query_sid = sids[query_idx]

            neighbor_indices, distances = find_nearest_neighbors(sids, query_sid, k=TOP_K)

            print(f"{'Rank':<5} | {'Hamming Distance':<18} | {'Title'}")
            print("-" * 50)
            for i, (idx, dist) in enumerate(zip(neighbor_indices, distances)):
                # The distance is a proportion; multiply by SID length for raw bit difference
                raw_dist = int(dist * sids.shape[1])
                print(f"{i+1:<5} | {dist:<18.4f} ({raw_dist} bits) | {movie_titles[idx]}")

if __name__ == '__main__':
    main()
