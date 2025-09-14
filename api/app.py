from fastapi import FastAPI
from data import preprocess_data,lemmetization,remove_stopwords,lower_case,remove_punctuation,remove_urls,remove_html_tags,remove_small_sentences

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(text: str):
    text=lemmetization(text)
    text=remove_stopwords(text)
    text=lower_case(text)
    text=remove_punctuation(text)
    text=remove_urls(text)
    text=remove_html_tags(text)
    text=remove_small_sentences(text)
    text=preprocess_data(text)
    
    return {"prediction": "positive"}