from tensorflow import keras

"""
Task: Distinguishing between samples from the model and samples from training data
TODO: crop the image to be 176 by 176
"""
class Discriminator:
    def __init__(self):
        self.createModel()

    """  
    Takes in images and outputs if it is generated or real!
    """
    def createModel(self):
        self.discriminator = keras.Sequential(
            [
                keras.Input(shape=(176, 176, 1)),
                
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.1),
                
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.1),
                
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.1),
                
                keras.layers.GlobalMaxPool2D(),
                keras.layers.Dense(1)
            ],
            name="discriminator",
        )
    
    def summary(self):
        self.discriminator.summary()