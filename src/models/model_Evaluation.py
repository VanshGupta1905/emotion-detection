import numpy as np
import pandas as pd
import logging
import json
import pickle
import mlflow
import dagshub
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

file_handler=logging.FileHandler('model_evaluation.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def setup_mlflow():
    """Set up MLflow tracking."""
    dagshub.init(repo_owner='VanshGupta1905', repo_name='emotion-detection', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/VanshGupta1905/emotion-detection.mlflow")
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.sklearn.autolog()

def load_model(model_path:str):
    """Load a model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model=pickle.load(f)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Unexpected error while loading model: %s', e)
        raise

def load_data(path)->pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df=pd.read_csv(path)
        logger.debug('Data loaded from %s', path)
        return df
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise


def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    """Evaluate a model."""
    try:
        y_pred=clf.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)
        metrics={'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}
        logger.debug('Model evaluated successfully')
        return metrics
    except Exception as e:
        logger.error('Unexpected error while evaluating model: %s', e)
        raise

def save_metrics(metrics:dict,metrics_path:str):
    """Save metrics to a JSON file."""
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f,indent=4)
        logger.debug('Metrics saved to %s', metrics_path)
    except Exception as e:
        logger.error('Unexpected error while saving metrics: %s', e)
        raise



def main():
    """Main function."""
    try:
        setup_mlflow()
        with mlflow.start_run():
            X_test=load_data('./data/processed/test_tfidf.csv')
            y_test=load_data('./data/interim/test.csv')['sentiment']
            model=load_model('./models/model.pkl')

            metrics=evaluate_model(model,X_test,y_test)
            mlflow.log_metrics(metrics)
            save_metrics(metrics,'./reports/metrics.json')
            logger.info('Model evaluation completed successfully')
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise e
    
if __name__=='__main__':
    main()