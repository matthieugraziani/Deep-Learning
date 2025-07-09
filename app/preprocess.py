import numpy as np
from PIL import Image
import io


def prepocess_image(image_bytes):
    # Convert the image to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image to 28x28 pixels
    image = image.resize((224, 224))
    
    # Convert the image to grayscale
    image_black = image.convert("L") # 'P' stands for RGB et 'L' pour "Luminance" (grayscale)
    
    # Convert the image to a numpy array
    image_array = np.array(image_black)
    
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    
    # Reshape the array to match the input shape of the model (1, 28, 28, 1)
    image_final = np.reshape(image_array, (1, 224, 224, 1))
    
    return image_final

def prepocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    
    # Split the text into words
    words = text.split()
    
    # Remove stop words (for simplicity, we will just remove common English stop words)
    stop_words = set(['the', 'is', 'in', 'and', 'to', 'a', 'of', 'that', 'it', 'with'])
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)