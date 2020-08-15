import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator
from gan import GAN, GANMonitor
from tensorflow import keras


HUMANS_DIR = os.path.join(os.getcwd(),"humans", "img_align_celeba")
HUMAN_PICS = os.listdir(HUMANS_DIR)
HEIGHT, WIDHT = (176, 176)

def prepare_image(filename, norm=True):
    # Open Image
    img_path = os.path.join(HUMANS_DIR, filename)
    img_gray = cv2.imread(img_path)

    # To grayscale
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

    # Crop
    orginal_height = img_gray.shape[0]
    orginal_width = img_gray.shape[1]
    height_margin = (orginal_height - HEIGHT)//2 
    width_margin = (orginal_width - WIDHT)//2
    img_gray = img_gray[height_margin:height_margin+HEIGHT, width_margin:width_margin+WIDHT]
    assert img_gray.shape == (HEIGHT, WIDHT)

    # Normalize
    img_gray = np.asarray(img_gray, dtype=np.float32) 
    img_gray /= 255

    # Reshape
    if norm:
        img_gray = keras.preprocessing.image.img_to_array(img_gray)
    return img_gray

def prepare_data(size=100, batch_size=30, prefetch=32):
    dataset=[]
    for i in range(size):
        dataset.append(prepare_image(HUMAN_PICS[i]))

    dataset = np.asarray(dataset, dtype=np.float32) 
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(buffer_size=size*2).batch(batch_size).prefetch(prefetch)

    return dataset


def train(dataset, epochs=30, num_images=1, latent_dim=256, learning_rate_g=0.00005, learning_rate_d=0.00005):
    discriminator = Discriminator().discriminator
    generator = Generator(latent_dim).generator

    gan = GAN(discriminator, generator, latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate_d),
        g_optimizer=keras.optimizers.Adam(learning_rate_g),
        loss_function=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    gan.fit(
        dataset, epochs=epochs, callbacks=[GANMonitor(num_images, latent_dim)]
    )
    
    # gan.save("gan_model.h5")
    generator.save("generator_model.h5")
    discriminator.save("discriminator_model.h5")

if __name__ == "__main__":
    # train(prepare_data())
    plt.imshow(prepare_image("000013.jpg", norm=False))
    plt.show()