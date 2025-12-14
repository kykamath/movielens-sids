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
        This can include base experiment names (e.g., 'mpnet-base') or their
        smoke test variants (e.g., 'mpnet-base-smoke-test').
        The script will load the corresponding model version for each experiment
        from the Hugging Face Hub (based on the version tag).

    Usage:
        # Compare the base mpnet and roberta-large models
        python evaluate.py --experiments mpnet-base roberta-large

        # Evaluate a smoke test version of the nemotron-8b model
        python evaluate.py --experiments nemotron-8b-smoke-test

        # Compare a full experiment with its smoke test counterpart
        python evaluate.py --experiments roberta-large roberta-large-smoke-test
"""

import argparse
import torch
import numpy as np
from scipy.spatial.distance import cdist

from pipeline import PipelineConfig, DatasetManager, EmbeddingManager, RQVAEOrchestrator
from experiments import EXPERIMENTS, ExperimentConfig

# --- Configuration ---
# A few interesting movies to test the semantic search on.
PROBE_MOVIES = ["Matrix, The (1999)", "Toy Story (1995)", "Godfather, The (1972)", "Pulp Fiction (1994)"]
TOP_K = 5 # Number of nearest neighbors to find

def find_nearest_neighbors(sids: np.ndarray, query_sid: np.ndarray, k: int):
    """Finds the top-k nearest neighbors using Hamming distance."""
    # cdist with 'hamming' calculates the proportion of differing bits
    distances = cdist(query_sid.reshape(1, -1), sids, metric='hamming')[0]
    # Get the indices of the top-k smallest distances
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices, distances[nearest_indices]

def main():
    # Generate all possible experiment choices for argparse
    all_experiment_choices = list(EXPERIMENTS.keys())
    for exp_name in list(EXPERIMENTS.keys()): # Iterate over a copy to modify
        all_experiment_choices.append(f"{exp_name}-smoke-test")
    if "smoke-test" not in all_experiment_choices:
        all_experiment_choices.append("smoke-test")


    parser = argparse.ArgumentParser(
        description="Compare semantic quality of models from different experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--experiments', 
        type=str, 
        nargs='+', # Accepts one or more experiment names
        required=True,
        choices=all_experiment_choices,
        help='A list of experiment names to compare. Can include smoke test variants.'
    )
    args = parser.parse_args()

    # --- 1. Load Data ---
    print("--- Loading source data... ---")
    base_config = PipelineConfig(enable_progress_bar=True) 
    dataset_manager = DatasetManager(base_config)
    source_dataset = dataset_manager.load_source_dataset()
    
    title_to_item = {item['title']: item for item in source_dataset}
    movie_titles = [item['title'] for item in source_dataset]

    # --- 2. Iterate Through Experiments and Evaluate ---
    for experiment_full_name in args.experiments:
        print(f"\n\n--- Evaluating Experiment: {experiment_full_name} ---")
        
        is_smoke_test_variant = experiment_full_name.endswith("-smoke-test")
        
        if is_smoke_test_variant:
            base_experiment_name = experiment_full_name.replace("-smoke-test", "")
            if base_experiment_name == "smoke-test":
                experiment_config_obj = None
            else:
                experiment_config_obj = EXPERIMENTS.get(base_experiment_name)
        else:
            base_experiment_name = experiment_full_name
            experiment_config_obj = EXPERIMENTS.get(base_experiment_name)

        # A. Configure and load the specific model for the experiment
        config = PipelineConfig(
            experiment_name=base_experiment_name,
            experiment_config=experiment_config_obj, 
            smoke_test=is_smoke_test_variant,
            enable_progress_bar=True # *** THIS IS THE FIX ***
        )
        config.experiment_name = experiment_full_name 

        model_orchestrator = RQVAEOrchestrator(config)

        # B. Generate embeddings using the *correct* model for this experiment
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
                raw_dist = int(dist * sids.shape[1])
                print(f"{i+1:<5} | {dist:<18.4f} ({raw_dist} bits) | {movie_titles[idx]}")

if __name__ == '__main__':
    main()
