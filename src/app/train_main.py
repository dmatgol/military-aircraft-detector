import logging

from pipelines import RunMode
from pipelines.inference import Inference
from pipelines.train import Train
from utils.cli import parse_cli_args
from utils.utils import read_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:

    args = parse_cli_args()
    model_config = read_model_config("configs/model.yaml")
    logging.info(f"Started application in {args.run_mode} mode.")
    args.run_mode = RunMode.TRAIN
    if args.run_mode == RunMode.TRAIN:
        Train(model_config=model_config).run()
    elif args.run_mode == RunMode.INFERENCE:
        Inference(model_config=model_config).run()
    else:
        raise ValueError("Provided command has not been implemented yet")


if __name__ == "__main__":
    main()
