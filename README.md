# Emotion Detection from Text

This project is a machine learning pipeline for detecting emotions from text data. It uses DVC for data versioning and experiment tracking, and MLflow for logging experiment runs and metrics.

## Project Structure

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VanshGupta1905/emotion-detection.git
    cd emotion-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Pipeline

This project uses DVC to manage the machine learning pipeline. To run the full pipeline, use the following command:

```bash
dvc repro
```

This command will execute the stages defined in `dvc.yaml` in the correct order:

1.  `data_ingestion`: Splits the raw data into training and testing sets.
2.  `data_preprocessing`: Cleans and preprocesses the text data.
3.  `feature_engineering`: Transforms the text data into numerical features using TF-IDF.
4.  `model_training`: Trains a RandomForestClassifier on the training data.
5.  `model_evaluation`: Evaluates the trained model on the testing data and logs metrics.

## Experiment Tracking

This project is integrated with DagsHub and MLflow for experiment tracking. When you run the pipeline, MLflow will automatically log the parameters, metrics, and model artifacts to the DagsHub repository.
