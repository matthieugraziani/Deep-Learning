import tensorflow as tf


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False) 
    return model

def predict_image(model, image):
    prediction = model.predict(image)
    return prediction.tolist()
