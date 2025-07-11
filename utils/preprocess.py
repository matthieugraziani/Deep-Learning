import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # 'L' = niveaux de gris
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if image_array.ndim == 2:  # niveaux de gris → RGB
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA → RGB
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)  # ajouter batch
    return image_array