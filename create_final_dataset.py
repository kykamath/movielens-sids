"""
create_final_dataset.py

This script orchestrates the creation of the final dataset with Semantic IDs (SIDs).
It supports using models from different experiments, a dummy mode, and a smoke test mode.

This script can be run from the command line with the following optional arguments:

--experiment [name]
    Description:
        Specifies which experiment's model to use for generating the dataset.
        The script will load the model version tagged with the experiment name.

    Usage:
        `python create_final_dataset.py --experiment roberta-large`

--dummy
    Description:
        Activates a "dummy run" mode for the quickest possible local test. It uses
        a minimal dataset, generates SIDs with a new untrained model, and DOES NOT
        upload the dataset.

    Usage:
        `python create_final_dataset.py --dummy`

--smoke-test
    Description:
        Activates a "smoke test" mode for a true end-to-end integration test.
        It uses a small dataset, loads the corresponding "-smoke-test" model from
        the Hub, and UPLOADS the resulting dataset.

    Usage:
        `python create_final_dataset.py --smoke-test`
        `python create_final_dataset.py --experiment roberta-large --smoke-test`

Note:
    `--dummy` and `--smoke-test` are mutually exclusive. `--dummy` takes precedence.
"""

import argparse
from pipeline import PipelineConfig, DatasetManager, EmbeddingManager, RQVAEOrchestrator
from experiments import EXPERIMENTS

def main():
    """
    Main workflow for creating the final dataset.
    1. Parses command-line arguments.
    2. Initializes configuration based on the selected mode.
    3. Runs the full pipeline.
    4. Creates and (optionally) uploads the final dataset.
    """
    parser = argparse.ArgumentParser(
        description="Create a dataset with SIDs using a model from a specified experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--experiment', 
        type=str, 
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help='The name of the experiment whose model should be used.'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Run in dummy mode for a quick local test (no uploads).'
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run in smoke test mode for a full end-to-end test (with uploads).'
    )
    args = parser.parse_args()

    if args.dummy and args.smoke_test:
        print("Warning: Both --dummy and --smoke-test were provided. --dummy takes precedence.")
        args.smoke_test = False

    experiment_config = EXPERIMENTS.get(args.experiment) if args.experiment else None

    config = PipelineConfig(
        experiment_name=args.experiment, 
        experiment_config=experiment_config, 
        dummy_run=args.dummy,
        smoke_test=args.smoke_test
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
