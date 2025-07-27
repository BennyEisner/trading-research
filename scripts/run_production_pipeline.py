#!/usr/bin/env python3

"""
Production Pipeline Runner - Industry Standard ML Pipeline
"""

import argparse
import sys
from pathlib import Path

sys.path.append(".")

from production_pipeline import ProductionMLPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Production ML Pipeline")
    parser.add_argument(
        "--config", default="production_config.json", help="Configuration file path (default: production_config.json)"
    )
    parser.add_argument("--quick-test", action="store_true", help="Use quick test configuration for faster iteration")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning stage")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["data", "tuning", "training", "evaluation", "artifacts"],
        help="Run only specific stages",
    )

    args = parser.parse_args()

    # Select configuration
    if args.quick_test:
        config_file = "quick_test_config.json"
        print("Running QUICK TEST pipeline")
    else:
        config_file = args.config
        print("Running FULL PRODUCTION pipeline")

    print(f"Configuration: {config_file}")

    # Validate config file exists
    if not Path(config_file).exists():
        print(f"Configuration file not found: {config_file}")
        return 1

    try:
        # Initialize pipeline
        pipeline = ProductionMLPipeline(config_file)

        # Modify configuration based on arguments
        if args.skip_tuning:
            pipeline.config["pipeline"]["stages"]["hyperparameter_tuning"] = False
            print("Hyperparameter tuning disabled")

        if args.stages:
            # Disable all stages first
            for stage in pipeline.config["pipeline"]["stages"]:
                pipeline.config["pipeline"]["stages"][stage] = False

            # Enable only requested stages
            stage_mapping = {
                "data": "data_preparation",
                "tuning": "hyperparameter_tuning",
                "training": "model_training",
                "evaluation": "model_evaluation",
                "artifacts": "model_artifacts",
            }

            for stage in args.stages:
                pipeline.config["pipeline"]["stages"][stage_mapping[stage]] = True

            print(f"Running only stages: {', '.join(args.stages)}")

        # Run the pipeline
        results = pipeline.run_full_pipeline()

        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        return 0

    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

