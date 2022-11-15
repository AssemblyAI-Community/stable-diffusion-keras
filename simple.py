from keras_cv.models import StableDiffusion
from PIL import Image

model = StableDiffusion()
img = model.text_to_image("Iron Man making breakfast")
Image.fromarray(img[0]).save("simple.png")