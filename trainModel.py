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

def prepare_image(filename, norm=True, ):
    # Open Image
    img_path = os.path.join(HUMANS_DIR, filename)
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # To grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop
    orginal_height = img.shape[0]
    orginal_width = img.shape[1]
    height_margin = (orginal_height - HEIGHT)//2 
    width_margin = (orginal_width - WIDHT)//2
    img = img[height_margin:height_margin+HEIGHT, width_margin:width_margin+WIDHT]
    assert img.shape == (HEIGHT, WIDHT, 3) # 3 channels

    # Normalize
    img = np.asarray(img, dtype=np.float32) 
    img /= 255

    # Reshape
    if norm:
        img = keras.preprocessing.image.img_to_array(img)
    return img

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
    pil_img = tf.keras.preprocessing.image.array_to_img(prepare_image("000013.jpg"))
    pil_img.save("test.png")
    # plt.imshow(pil_img)
    # plt.show()

    discriminator = Discriminator()
    generator = Generator(256)

    print(discriminator.summary())
    print(generator.summary())