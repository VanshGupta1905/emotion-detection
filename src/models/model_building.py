from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import logging
import os
import yaml
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
import pickle
logger=logging.getLogger('model_building')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

file_handler=logging.FileHandler('model_building.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def setup_mlflow():
    """Set up MLflow tracking."""
    dagshub.init(repo_owner='VanshGupta1905', repo_name='emotion-detection', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/VanshGupta1905/emotion-detection.mlflow")
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.sklearn.autolog() # This can cause issues with DagsHub when logging models manually


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(path)->pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df=pd.read_csv(path)
        logger.debug('Data loaded from %s', path)
        return df
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def train_model(X_train:pd.DataFrame,y_train:pd.DataFrame,params:dict)->RandomForestClassifier:
    """Train a Random Forest Classifier."""
    try:
        model=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=42)
        model.fit(X_train,y_train)
        logger.debug('Model trained successfully')
        return model
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def save_model(model:RandomForestClassifier,model_path:str)->None:
    """Save a Random Forest Classifier."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug('Model saved to %s', model_path)
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def main():
    """Main function."""
    try:
        setup_mlflow()
        
        # Initialize DagsHub for MLflow tracking
        # dagshub.init(repo_owner='VanshGupta1905', repo_name='emotion-detection', mlflow=True)
        # mlflow.set_tracking_uri("https://dagshub.com/VanshGupta1905/emotion-detection.mlflow")

        mlflow.set_experiment("emotion_detection_experiment")
        with mlflow.start_run():
            params = load_params('params.yaml')['model_building']
            mlflow.log_params(params)
            
            # Load processed features
            X_train = load_data('./data/processed/train_tfidf.csv')
            
            # Load original training data to get the target variable
            train_original = load_data('./data/interim/train.csv')
            y_train = train_original['sentiment']
            
            model = train_model(X_train, y_train, params)
            # infer_signature=mlflow.models.infer_signature(X_train,y_train)
            # mlflow.sklearn.log_model(
            #     sk_model=model,
            #     artifact_path="random_forest_model",
            #     signature=infer_signature
            # )
            save_model(model, './models/model.pkl')
            
            logger.info('Model training completed successfully')
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise e

if __name__ == '__main__':
    main()