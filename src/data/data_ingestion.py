import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging

logger=logging.getLogger(__name__)
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

file_handler=logging.FileHandler('data_ingestion.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def load_data(path)->pd.DataFrame:
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise e
def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """ Preprocess the data """
    try:
        logger.info(f"Preprocessing data")
        df=df.drop(columns=['tweet_id'])
        final_df=df[df['sentiment'].isin(['happiness','sadness'])]
        final_df['sentiment']=final_df['sentiment'].map({'happiness':1,'sadness':0})
        logger.info(f"Data preprocessed successfully")
        return final_df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise e

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str):
    """ Save the data """
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.info(f"Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise e

def main():
    """ Main function """
    try:
       df=load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/refs/heads/main/tweet_emotions.csv')
       df=preprocess_data(df)
       train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)
       save_data(train_data,test_data,data_path='./data')
       logger.info(f"Data ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise e

if __name__ == '__main__':
    main()