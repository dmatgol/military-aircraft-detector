<div align="center">
<img src=https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.neweurope.eu%2Fwp-content%2Fuploads%2F2018%2F06%2Fh_51519413.jpg&imgrefurl=https%3A%2F%2Fwww.neweurope.eu%2Farticle%2Fnato-planes-begin-patrolling-montenegros-skies%2F&tbnid=YUUlpIPFcVC3dM&vet=12ahUKEwi5gsi9ipD6AhXEqqQKHW9zCiYQMygOegUIARDbAQ..i&docid=7UMjctyRWQ1KrM&w=3543&h=2362&q=aircraft%20nato&ved=2ahUKEwi5gsi9ipD6AhXEqqQKHW9zCiYQMygOegUIARDbAQ" width="150px"/><br/>

# Nato Aircraft Classifier
[![CI](https://github.com/dmatgol/nato-ac-classifier/actions/workflows/main.yml/badge.svg)](https://github.com/dmatgol/nato-ac-classifier/actions/workflows/main.yml)
</div>

Goal: Build an AI solution to quickly and efficiently identify non-NATO military aircrafts in images.

# Instalation

**Option 1: virtualenv**
1. make sure pip3 is installed ( ubuntu machine)
```
apt update -y
apt install python3-pip -y
```
2. install virtualenv
```
pip3 install virtualenv
```
3. Create an environment for the project
```
virtualenv -p python3 ~/virtualenvs/non_nato_aircraft_classifier
```
4. Activate the virtualenv
```
source ~/virtualenvs/non_nato_classifier/bin/activate
```
5. Install requirements on your virtualenv from requirements.txt
```
pip install -r requirements.txt
```

# Repo Structure

This repository contains the following structure:
 - configs:
    - dataset:
        - train_images_path: Path of the folder containing the images in the dataset used for training. These will be split into: train, validation and test set. 
        - labels_path: Path of the folder containing the labels.
        - test_images_path: Path of the folder containing new images not used in the train process. This path is used if we want to test the model in a completely new dataset. Only used when running `main.py` in inference mode (attribute: `run_mode: serve` in `configs/model/cnn.yaml`).
    - model: Contains the model hyperparameters (The parameters in this file correspond to the parameters of the best performing model achieved.)
- dataset:
    - Test: Split in non_nato and positive_nato folders contains the images used in the test set (model metrics are reported based on this test set).
    - Train: Split in non_nato and positive_nato folders contains the images used in the train dataset.
    - Val: Split in non_nato and positive_nato folders contains the images used in the validation dataset.
- model_results:
    - Best_trained_model: Folder containing the best performing model.
    - Serve: Folder containing the results when ```main.py``` is run in `serve` (inference) mode. For every time, that ```main.py``` is run in `serve` mode, a new folder under **Serve**  will be created (folder name will be the timestamp when the `main.py` was started) to save the results for different runs of `main.py`.
    - Train: Same logic as above, but when `main.py` is run in training mode.
- src: 
    - Contains the model creation, training, evaluation and prediction mode. Additionally, it also contains the metrics used to evaluate the model performance.
- utils:
    - A set of useful functions, including constants definition.

# Running locally

1. Run the best performing model in inference mode for unseen images/ new dataset:
    - Specify `test_images_path` in `configs/dataset/data_path.yaml`.
    - Ensure `run_mode` is set to `serve` in `configs/model/cnn.yaml`.
    - Run:
    ```python main.py```
    - Output will be saved in a json containing the following format:
    ```
    {image_file}: {'Prediction_probability': value, Predicted Class: value (Nato/Non Nato)}
    ```
    - Results will be store under `{your_current_dir}/model_results/serve/{timestamp}/`.
2. Evaluate best performing model on test dataset but skipping the train step:
    - Specify the correct path for the images folder and labels folder in `configs/dataset/data_path.yaml`.
    - Ensure `run_mode` is set to `train` in `configs/model/cnn.yaml`.
    - Ensure that the argument `skip_train` is set to `True` in the `configs/model/cnn.yaml`.
    - Run:
    ```python main.py```
    - Results will be store under `{your_current_dir}/model_results/train/{timestamp}/`.

3. Train a new model and evaluate its performance on the testset:
    - Specify the correct path for the images folder and labels folder in `configs/dataset/data_path.yaml`.
    - Ensure `run_mode` is set to `train` in `configs/model/cnn.yaml`.
    - Change the model hyperparameters in `configs/dataset/cnn.yaml` (optional).
    - Run:
    ```python main.py```
    - Results will be store under `{your_current_dir}/model_results/train/{timestamp}/`.

**Note**: Default configuration is the option 2.