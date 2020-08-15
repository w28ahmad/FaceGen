from tensorflow import keras

"""  
Task: Confuse the discriminator & generate realistic faces
@param latent_dim: latent input space of the generator,
                   learns the best latent space for generating faces
"""
class Generator:
    def __init__(self, latent_dim=256):
        self.latent_dim = latent_dim
        self.createModel()

    def createModel(self):
        self.generator = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim, )),
                keras.layers.Dense(9 * 9 * self.latent_dim),

                keras.layers.LeakyReLU(alpha=0.3),
                keras.layers.Reshape(target_shape=(9, 9, self.latent_dim)),
            
                keras.layers.Conv2DTranspose(self.latent_dim, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.3),
            
                keras.layers.Conv2DTranspose(self.latent_dim, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.3),
            
                keras.layers.Conv2DTranspose(self.latent_dim, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.3),
            
                keras.layers.Conv2DTranspose(self.latent_dim, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.3),

                keras.layers.Conv2D(3, (9, 9), padding="same", activation="sigmoid")
            ],
            name="generator",
        )

    def summary(self):
        self.generator.summary()