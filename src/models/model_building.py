import numpy as np
import pandas as pd
import logging
import os
import yaml
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
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
        params = load_params('params.yaml')['model_building']
        
        # Load processed features
        X_train = load_data('./data/processed/train_tfidf.csv')
        
        # Load original training data to get the target variable
        train_original = load_data('./data/interim/train.csv')
        y_train = train_original['sentiment']
        
        model = train_model(X_train, y_train, params)
        save_model(model, './models/model.pkl')
        
        logger.info('Model training completed successfully')
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise e

if __name__ == '__main__':
    main()