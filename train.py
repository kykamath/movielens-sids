"""
train.py

This script orchestrates the training of the RQ-VAE model using the modular,
OOP-based pipeline. It supports running different experiments and a dummy mode for quick tests.

This script can be run from the command line with the following optional arguments:

--experiment [name]
    Description:
        Specifies which experiment configuration to run. The available experiments
        are defined in the `experiments.py` file. Each experiment can have its own
        embedding model, hyperparameters, and Hugging Face Hub model ID.

    Usage:
        `python train.py --experiment roberta-large`

    If this argument is not provided, the script will run using the default
    parameters defined in `PipelineConfig`.

--dummy
    Description:
        Activates a "dummy run" mode. This is used for quickly testing the end-to-end
        pipeline without waiting for a full training cycle. When this flag is set,
        the script will:
        - Use a very small subset of the data.
        - Train for only one epoch.
        - Skip uploading the final model to the Hugging Face Hub.
    
    This flag can be combined with `--experiment` to test a specific experiment's
    configuration in dummy mode.

    Usage:
        `python train.py --dummy`
        `python train.py --experiment roberta-large --dummy`
"""

import argparse
from pipeline import PipelineConfig, RQVAEOrchestrator
from experiments import EXPERIMENTS

def main():
    """
    Main training workflow.
    1. Parses command-line arguments for experiment name and dummy run mode.
    2. Loads the appropriate configuration.
    3. Initializes the RQ-VAE orchestrator.
    4. Starts the training process.
    """
    parser = argparse.ArgumentParser(
        description="Train an RQ-VAE model with a specified experiment configuration.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve formatting of help messages
    )
    parser.add_argument(
        '--experiment', 
        type=str, 
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help='The name of the experiment to run, as defined in experiments.py.'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='If set, runs in dummy mode on a small subset of data for a quick end-to-end test.'
    )
    args = parser.parse_args()

    # Get the experiment config if an experiment is specified
    experiment_config = EXPERIMENTS.get(args.experiment) if args.experiment else None

    # Initialize configuration. The PipelineConfig class now handles all the logic
    # for layering the default, experiment, and dummy configurations.
    config = PipelineConfig(
        experiment_name=args.experiment, 
        experiment_config=experiment_config, 
        dummy_run=args.dummy
    )
    
    orchestrator = RQVAEOrchestrator(config)
    orchestrator.train()
    
    print("\n--- Training Script Finished ---")

if __name__ == '__main__':
    main()
