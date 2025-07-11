import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False) 
    return model