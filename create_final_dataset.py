"""
create_final_dataset.py

This script orchestrates the creation of the final dataset with Semantic IDs (SIDs)
using the modular, OOP-based pipeline. It supports using models from different experiments
and a dummy mode for quick tests.

This script can be run from the command line with the following optional arguments:

--experiment [name]
    Description:
        Specifies which experiment's model to use for generating the dataset.
        The script will load the model version tagged with the experiment name from
        the Hugging Face Hub. The available experiments are defined in `experiments.py`.

    Usage:
        `python create_final_dataset.py --experiment roberta-large`

    If this argument is not provided, the script will use the model from the `main`
    branch of the default Hub repository.

--dummy
    Description:
        Activates a "dummy run" mode for quick end-to-end testing. When this flag
        is set, the script will:
        - Use a very small subset of the data.
        - Generate SIDs using a new, untrained model (skipping the download).
        - Skip uploading the final dataset to the Hugging Face Hub.

    This flag can be combined with `--experiment` to test a specific experiment's
    configuration in dummy mode.

    Usage:
        `python create_final_dataset.py --dummy`
        `python create_final_dataset.py --experiment roberta-large --dummy`
"""

import argparse
from pipeline import PipelineConfig, DatasetManager, EmbeddingManager, RQVAEOrchestrator
from experiments import EXPERIMENTS

def main():
    """
    Main workflow for creating the final dataset.
    1. Parses command-line arguments for experiment name and dummy run mode.
    2. Initializes configuration based on the selected settings.
    3. Runs the full pipeline: load data, generate embeddings, load model, generate SIDs.
    4. Creates and (optionally) uploads the final dataset.
    """
    parser = argparse.ArgumentParser(
        description="Create a dataset with SIDs using a model from a specified experiment.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve formatting of help messages
    )
    parser.add_argument(
        '--experiment', 
        type=str, 
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help='The name of the experiment whose model should be used, as defined in experiments.py.'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='If set, runs in dummy mode on a small subset of data for a quick end-to-end test.'
    )
    args = parser.parse_args()

    # Get the experiment config if an experiment is specified
    experiment_config = EXPERIMENTS.get(args.experiment) if args.experiment else None

    # Initialize configuration and managers. The PipelineConfig class handles all logic.
    config = PipelineConfig(
        experiment_name=args.experiment, 
        experiment_config=experiment_config, 
        dummy_run=args.dummy
    )
    dataset_manager = DatasetManager(config)
    embedding_manager = EmbeddingManager(config)
    model_orchestrator = RQVAEOrchestrator(config)

    try:
        source_dataset = dataset_manager.load_source_dataset()
        embeddings_tensor, items_with_plots_indices, embeddings_list = embedding_manager.generate(source_dataset)
        rq_model = model_orchestrator.load_from_hub()
        sids = model_orchestrator.generate_sids(rq_model, embeddings_tensor)
        final_dataset = dataset_manager.create_final_dataset(
            source_dataset, 
            embeddings_list, 
            sids, 
            items_with_plots_indices
        )
        dataset_manager.push_to_hub(final_dataset)

    except (ValueError, RuntimeError) as e:
        print(f"\n❌ An error occurred: {e}")
        return

    print("\n--- Dataset Creation Script Finished ---")

if __name__ == '__main__':
    main()
