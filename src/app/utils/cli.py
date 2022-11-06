import argparse

from pipelines import RunMode


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_mode",
        dest="run_mode",
        help="Select the run mode: train or inference.",
        default="inference",
    )

    args = parser.parse_args()

    # Validate subparser arguments
    if args.run_mode not in [
        RunMode.TRAIN,
        RunMode.INFERENCE,
    ]:
        parser.error(
            "--run_mode not available. Please choose either: train or inference."
        )

    return args
