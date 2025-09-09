import pandas as pd
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

file_handler=logging.FileHandler('feature_engineering.log')
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
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise e
    
def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int=500)->tuple:
    try:
        logger.info(f"Applying TF-IDF to data")
        tfidf=TfidfVectorizer(max_features=max_features)
        
        X_train_transformed = tfidf.fit_transform(train_data['content'].astype(str))
        X_test_transformed = tfidf.transform(test_data['content'].astype(str))
        
        feature_names = tfidf.get_feature_names_out()
        
        X_train = pd.DataFrame(X_train_transformed.toarray(), columns=feature_names)
        X_test = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)

        logger.info(f"TF-IDF applied to data successfully")
        return X_train,X_test

    except Exception as e:
        logger.error(f"Error applying TF-IDF to data: {e}")
        raise e
    
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train.csv')
        test_data = load_data('./data/interim/test.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()