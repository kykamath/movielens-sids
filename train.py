"""
train.py

This script orchestrates the training of the RQ-VAE model using the modular,
OOP-based pipeline. It supports running different experiments, a dummy mode, and a smoke test mode.

This script can be run from the command line with the following optional arguments:

--experiment [name]
    Description:
        Specifies which experiment configuration to run. The available experiments
        are defined in the `experiments.py` file.

    Usage:
        `python train.py --experiment roberta-large`

--dummy
    Description:
        Activates a "dummy run" mode for the quickest possible local test. It uses
        a minimal dataset, runs for one epoch, and DOES NOT upload anything.
        This is for checking if the code runs without syntax errors.

    Usage:
        `python train.py --dummy`

--smoke-test
    Description:
        Activates a "smoke test" mode for a true end-to-end integration test.
        It uses a small dataset, runs for a few epochs, and UPLOADS the resulting
        model to a separate "-smoke-test" repository on the Hub. This verifies
        the entire pipeline, including authentication and uploads.

    Usage:
        `python train.py --smoke-test`
        `python train.py --experiment roberta-large --smoke-test`

Note:
    `--dummy` and `--smoke-test` are mutually exclusive. `--dummy` takes precedence.
"""

import argparse
from pipeline import PipelineConfig, RQVAEOrchestrator
from experiments import EXPERIMENTS

def main():
    """
    Main training workflow.
    1. Parses command-line arguments.
    2. Loads the appropriate configuration based on the selected mode.
    3. Initializes the RQ-VAE orchestrator.
    4. Starts the training process.
    """
    parser = argparse.ArgumentParser(
        description="Train an RQ-VAE model with a specified experiment configuration.",
        formatter_class=argparse.RawTextHelpFormatter
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
    
    orchestrator = RQVAEOrchestrator(config)
    orchestrator.train()
    
    print("\n--- Training Script Finished ---")

if __name__ == '__main__':
    main()
