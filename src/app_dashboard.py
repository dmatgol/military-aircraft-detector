from frontend.app_design import app_design
from utils.utils import read_model_config


def main():
    model_config = read_model_config("configs/model.yaml")

    app_design(model_config)


if __name__ == "__main__":
    main()
