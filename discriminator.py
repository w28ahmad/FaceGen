from tensorflow import keras

"""
Task: Distinguishing between samples from the model and samples from training data
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
                keras.Input(shape=(144, 144, 1)),
                
                keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.LeakyReLU(alpha=0.4),
                
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.LeakyReLU(alpha=0.4),
                
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.LeakyReLU(alpha=0.4),
                
                # keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                # keras.layers.MaxPooling2D(pool_size=2),
                # keras.layers.LeakyReLU(alpha=0.4),
                
                # keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                # keras.layers.MaxPooling2D(pool_size=2),
                # keras.layers.LeakyReLU(alpha=0.4),
                
                # keras.layers.GlobalMaxPool2D(),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Flatten(),

                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dropout(0.2),
             
                keras.layers.Dense(1),
            ],
            name="discriminator",
        )
    
    def summary(self):
        self.discriminator.summary()