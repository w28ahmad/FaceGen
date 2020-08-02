from discriminator import Discriminator
from generator import Generator

if __name__ == "__main__":
    latent_dim=256
    discriminator = Discriminator()
    generator = Generator(latent_dim)
