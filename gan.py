import tensorflow as tf
from tensorflow import keras

""" 
Credit: https://keras.io/examples/generative/dcgan_overriding_train_step/
"""
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_function):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_function = loss_function

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        
        # Sample the latent space randomly
        batch_size = tf.shape(real_images)[0]
        random_latent_vector = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Create fake images
        generated_images = self.generator(random_latent_vector)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Create and Combine labels
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.0002 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_function(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))
        # misleading_labels =  0.001 * tf.random.uniform(tf.shape(misleading_labels))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_function(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


    def predict(self, num_images=1, save=False):
        random_latent_vectors = tf.random.normal(shape=(num_images, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        imgs = []

        for i in range(num_images):
            imgs.append(keras.preprocessing.image.array_to_img(generated_images[i]))
            if save:
                imgs[i].save("./output/prediction_img_{i}.png".format(i=i))
        
        return imgs
            

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_images=5, latent_dim=256):
        self.num_images = num_images
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_images, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255

        generated_images.numpy()
        for i in range(self.num_images):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            # print(img)
            # keras.preprocessing.image.save_img(os.join(os.getcwd(), "output", "generated_img_{i}_{epoch}.png".format(i, epoch)), img)
            if epoch % 50 == 0:
              print(epoch, " Ended. Generating {image} picture".format(image=self.num_images))
              filename = "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch)
              img.save(filename)