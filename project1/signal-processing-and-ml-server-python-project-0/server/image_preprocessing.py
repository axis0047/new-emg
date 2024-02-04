from PIL import Image
import numpy as np

# Preprocess the image
def preprocess_image(image_path, target_size=(28, 28)):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize(target_size, Image.LANCZOS)
    return np.expand_dims(np.array(image), axis=-1)
