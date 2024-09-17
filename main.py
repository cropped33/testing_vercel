from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from typing import Dict

app = FastAPI()

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

class TextInput(BaseModel):
    text: str

@app.post("/api/predict")
async def predict(input_data: TextInput) -> Dict[str, float]:
    try:
        vectorized_text = vectorizer.transform([input_data.text])
        prediction = model.predict(vectorized_text)
        probabilities = model.predict_proba(vectorized_text)

        result = {
            'prediction': int(prediction[0]),
            'spam_probability': float(probabilities[0][1]),
            'ham_probability': float(probabilities[0][0])
        }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
