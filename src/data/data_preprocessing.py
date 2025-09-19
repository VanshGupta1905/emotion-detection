import string
import pandas as pd
import numpy as np
import os
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

file_handler=logging.FileHandler('data_ingestion.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def lemmetization(text:str)->str:
    """ Lemmetization of Text"""
    lemmatizer=WordNetLemmatizer()
    text=text.split()
    text=[lemmatizer.lemmatize(word) for word in text]
    text=' '.join(text)
    return text

def remove_stopwords(text:str)->str:
    """ Remove stopwords from Text"""
    stop_words=set(stopwords.words('english'))
    text=text.split()
    text=[word for word in text if word not in stop_words]
    text=' '.join(text)
    return text

def lower_case(text:str)->str:
    """ Convert text to lower case"""
    text=text.split()
    text=[word.lower() for word in text]
    text=' '.join(text)
    return text


def remove_punctuation(text:str)->str:
    """ Remove punctuation from Text"""
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=text.replace('؛','')
    text=re.sub('\s+',' ',text).strip()
    return text

def remove_urls(text:str)->str:
    """ Remove urls from Text"""
    text=re.sub(r'http\S+','',text)
    text=re.sub(r'www\S+','',text)
    return text

def remove_html_tags(text:str)->str:
    """ Remove html tags from Text"""
    text=re.sub(r'<.*?>','',text)
    return text

def remove_small_sentences(df:pd.DataFrame)->pd.DataFrame:
    for i in range(len(df)):
        if len(df.content.iloc[i].split())<3:
            df.content.iloc[i]=np.nan
    return df

def normalize_text(df:pd.DataFrame)->pd.DataFrame:
    df['content']=df['content'].apply(lambda x: remove_urls(x))
    df['content']=df['content'].apply(lambda x: remove_html_tags(x))
    df['content']=df['content'].apply(lambda x: remove_punctuation(x))
    df['content']=df['content'].apply(lambda x: lower_case(x))
    df['content']=df['content'].apply(lambda x: remove_stopwords(x))
    df['content']=df['content'].apply(lambda x: lemmetization(x))
    return df

def main():
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        train_data=normalize_text(train_data)
        test_data=normalize_text(test_data)
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)
        train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'),index=False)
        logger.info(f"Data preprocessed successfully")
        return train_data,test_data
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise e

if __name__ == '__main__':
    main()
