from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np

from model import load_model, predict_image
from preprocess import prepocess_image

class Prediction(BaseModel):
    Number: int
    Proba: dict

app = FastAPI()

model = load_model('./models/cnn_model_resnet50.keras')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification API"}

@app.post("/predict_image",response_model=Prediction)
async def predict_image_endpoint(file: UploadFile = File(...)):
    try : 
        image = await file.read()
        preprocessed_image = prepocess_image(image)
        prediction = predict_image(model, preprocessed_image)
        round_proba = {i : round(p, 5) for i, p in enumerate(prediction[0])}

        return {
            "Number": np.argmax(prediction[0]),
            "Proba" : round_proba
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))