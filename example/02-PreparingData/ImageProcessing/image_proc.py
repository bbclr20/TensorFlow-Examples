import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
"""My jupyter is temporary broken..."""

def display_image(image: np.array, figure_name: str):
    plt.figure(figure_name)
    plt.imshow(image)

sess = tf.InteractiveSession()

# read image with GFile 
with tf.gfile.GFile("../../../data/images/lena.jpg", "rb") as file:
    image_raw = file.read()
    image_tensor_512 = tf.image.decode_jpeg(image_raw)    # RGB image
    print("image_tensor.dtype: ", image_tensor_512.dtype) # tf.uint8
    image = sess.run(image_tensor_512)
    display_image(image, "Lena 512x512")
    
# write image with GFile
with tf.gfile.GFile("./leana_300.jpg", "wb") as file:
    image_tensor = tf.image.convert_image_dtype(image_tensor_512, tf.float32)
    image_tensor_300 = tf.image.resize_images(image_tensor, [300, 300], method=0)
    image_tensor_300 = tf.image.convert_image_dtype(image_tensor_300, tf.uint8)

    encoded_image = tf.image.encode_jpeg(image_tensor_300)
    file.write(sess.run(encoded_image))

    image = cv2.imread("./leana_300.jpg")            # cv2 return a BGR image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    display_image(image, "Lena 300x300")

# 600 is larger than 512, pad the image with 0
image_tensor_600 = tf.image.resize_image_with_crop_or_pad(image_tensor_512, 600, 600)
image = sess.run(image_tensor_600)
display_image(image, "Lena 600x600")

# 400 is smaller than 512, just crop the image   
image_tensor_400 = tf.image.resize_image_with_crop_or_pad(image_tensor_512, 400, 400)
image = sess.run(image_tensor_400)
display_image(image, "Lena 400x400")

# central crop with specific ratio
image_tensor_256 = tf.image.central_crop(image_tensor_512, 0.5)
image = sess.run(image_tensor_256)
display_image(image, "Lena 256X256")

# flip the image up to down
image_tensor_ud = tf.image.flip_up_down(image_tensor_512)
# image_tensor_ud = tf.image.random_flip_up_down(image_tensor_512)
image = sess.run(image_tensor_ud)
display_image(image, "Lena up to down")

# adjust brightness
image_tensor_adjb = tf.image.adjust_brightness(image_tensor_512, -0.5)
# image_tensor_adjb = tf.image.random_brightness(image_tensor_512, -0.5)
image = sess.run(image_tensor_adjb)
display_image(image, "Lena adjust brightness")

# adjust contrast
image_tensor_adjc = tf.image.adjust_contrast(image_tensor_512, -0.5)
# image_tensor_adjc = tf.image.random_contrast(image_tensor_512, -0.5)
image = sess.run(image_tensor_adjc)
display_image(image, "Lena adjust contrast")

# adjust hue
image_tensor_adjh = tf.image.adjust_hue(image_tensor_512, 0.3)
image = sess.run(image_tensor_adjh)
display_image(image, "Lena adjust hue")

# adjust saturation
image_tensor_adjs = tf.image.adjust_saturation(image_tensor_512, -5)
image = sess.run(image_tensor_adjs)
display_image(image, "Lena adjust saturation")

plt.show()
sess.close()
