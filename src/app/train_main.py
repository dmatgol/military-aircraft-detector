import logging

from pipelines.preprocessing import PreProcessing
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
    if args.run_preprocessing:
        logging.info("Started PreProcessing Step: Splitting the Dataset...")
        PreProcessing(args.full_dataset_dir).run()
    logging.info("Started training the model")
    Train(model_config=model_config).run()


if __name__ == "__main__":
    main()
