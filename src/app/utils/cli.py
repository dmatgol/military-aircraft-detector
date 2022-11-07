import argparse


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full_dataset_dir",
        dest="full_dataset_dir",
        help="Please provide the name of the directory where the full dataset is stored.",
    )

    parser.add_argument(
        "--run_preprocessing",
        dest="run_preprocessing",
        help="Please specify if you want to run the data preprocessing. Data Preprocessing step \
            will execute the train and test split. You can skip these if you have already done it.",
        default=True,
    )

    args = parser.parse_args()

    return args
