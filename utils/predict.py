import numpy as np

def predict_image(model, processed_image):
    predictions = model.predict(processed_image)
    classe_predite = np.argmax(predictions, axis=1)
    return predictions, classe_predite[0]