from fastapi import FastAPI, Request
from src.data import preprocess_data,lemmetization,remove_stopwords,lower_case,remove_punctuation,remove_urls,remove_html_tags,remove_small_sentences
import pickle
from contextlib import asynccontextmanager
import uvicorn
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models and categories into app.state on startup
    with open('./models/tfidf.pkl','rb') as f:
        app.state.tfidf = pickle.load(f)
    with open('./models/model.pkl','rb') as f:
        app.state.model = pickle.load(f)
    app.state.categories = {1:'happiness',0:'sadness'}
    yield
    # Clean up resources on shutdown (optional)

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(text: str, request: Request):
    # Preprocess input text
    processed_text = lemmetization(text)
    processed_text = remove_stopwords(processed_text)
    processed_text = lower_case(processed_text)
    processed_text = remove_punctuation(processed_text)
    processed_text = remove_urls(processed_text)
    processed_text = remove_html_tags(processed_text)
    
    # Use models from app state
    vectorized_text = request.app.state.tfidf.transform([processed_text])
    prediction = request.app.state.model.predict(vectorized_text)

    # Map prediction to category name
    category = request.app.state.categories[int(prediction[0])]
    
    return {"prediction": category}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)