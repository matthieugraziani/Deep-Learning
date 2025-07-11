from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
from model import load_model
from predict import predict_image
from preprocess import preprocess_image

class Prediction(BaseModel):
    Number: int
    Proba: dict

app = FastAPI()

model = load_model('./models/model_cnn_seq.keras')

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API FastAPI !"}

@app.post("/predict_image",response_model=Prediction)
async def predict_image_endpoint(file: UploadFile = File(...)):
    try : 
        image_bytes = await file.read()
        preprocessed_image = preprocess_image(image_bytes)
        prediction = predict_image(model, preprocessed_image)
        round_proba = {i : round(float(p), 5) for i, p in enumerate(prediction[0])}

        return {
            "Number": np.argmax(prediction[0]),
            "Proba" : round_proba
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))