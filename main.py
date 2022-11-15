from keras_cv.models import StableDiffusion
import requests
from PIL import Image
import numpy as np
import time

# Set a random seed
SEED = 119

########################################################################################################################
########################################### IMAGE GENERATION ###########################################################
########################################################################################################################

# Load the Stable Diffusion model from Keras. Read more about JIT compilation: https://www.assemblyai.com/blog/why-you-should-or-shouldnt-be-using-jax-in-2022/#just-in-time-compilation-with-jit
model = StableDiffusion(img_height=512, img_width=512, jit_compile=False)

# Set options for the image generation
options = dict(
    prompt="An alien riding a skateboard in space, vaporwave aesthetic, trending on ArtStation ",
    batch_size=1,
    num_steps=25,
    unconditional_guidance_scale=7,
    seed=SEED
)

# Run the generation
s = time.time()
img = model.text_to_image(**options)
print(f"Generation time: {time.time() - s} seconds")

# Save image
Image.fromarray(img[0]).save("generated.png")

########################################################################################################################
########################################### IMAGE INPAINTING ###########################################################
########################################################################################################################

# Fetch the image and save it as `main-on-skateboard.jpg`
file_URL = "https://c0.wallpaperflare.com/preview/87/385/209/man-riding-on-the-skateboard-photography.jpg"
r = requests.get(file_URL)
with open("man-on-skateboard.jpg", 'wb') as f:
    f.write(r.content)

# ---- CROP IMAGE ----
# Define crop settings
x_start = 80  # Starting x coordinate from the left of the image
width = 512
y_start = 0  # Starting y coordinate from the BOTTOM of the image
height = 512

# Load image as tensor
im = Image.open("man-on-skateboard.jpg")
img = np.array(im)

# Execute the crop and save
img = img[im.height-height-y_start:im.height-y_start, x_start:x_start+width]
new_filename = "man-on-skateboard-cropped.png"
Image.fromarray(img).save(new_filename)

# ---- MASK CREATION ----
# Define mask settings
x_start = 134
x_end = 374
y_start = 0
y_end = 369

# Open cropped image and load as a tensor
im = Image.open("man-on-skateboard-cropped.png")
img = np.array(im)

# Intiialize an empty mask
mask = np.ones((img.shape[:2]))

# Specify the masked region
mask[img.shape[0]-y_start-y_end:img.shape[1]-y_start, x_start:x_end] = 0

# ---- INPAINTING ----
# Add batchs dims
mask = np.expand_dims(mask, axis=0)
img = np.expand_dims(img, axis=0)

# Set inpainting options
inpaint_options = dict(
        prompt="A golden retriever on a skateboard",
        image=img,
        mask=mask,
        num_resamples=5,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=8.5,
        diffusion_noise=None,
        seed=SEED,
        verbose=True,
)

# Instantiate the model
inpainting_model = StableDiffusion(img_height=img.shape[1], img_width=img.shape[2], jit_compile=False)

# Run the inpainting
s = time.time()
inpainted = inpainting_model.inpaint(**inpaint_options)
print(f"Generation time: {time.time() - s} seconds")

# Save the inpainted image
Image.fromarray(inpainted[0]).save("inpainted.png")
