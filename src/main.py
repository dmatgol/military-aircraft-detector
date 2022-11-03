from pipelines.train import Train
from src.utils.utils import read_model_config


def main() -> None:

    model_config = read_model_config("configs/model.yaml")
    Train(model_config=model_config).run()


if __name__ == "__main__":
    main()
