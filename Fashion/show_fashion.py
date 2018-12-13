import os
import numpy as np
import string
from tqdm import tqdm
import matplotlib.pyplot as plt


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("models"):
    os.mkdir("models")

K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)


generator = Sequential()
generator.add(Dense(128*8*8, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 8, 8)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(3, kernel_size=(3, 3), padding='same', activation='tanh'))

generator.compile(loss='binary_crossentropy', optimizer=adam)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# X_train = (X_train.astype(np.float32) - 127.5)/127.5
# X_train = X_train[:, np.newaxis, :, :]
# ones_train  = X_train[np.where(y_train == 1)[0]]
ones_train = (X_train.astype(np.float32) - 127.5)/127.5

def plotDataSetImages(examples=100, dim=(10, 10), figsize=(16, 16)):
    image = ones_train[:100]
    plt.figure(figsize=figsize)
    for i in range(image.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(image[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_dataset_image_100.png')


# Create a wall of generated Cifar10 images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(12, 12) ,single = False):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    if single:
        generator.load_weights('./models/singleModel/dcgan_generator_epoch_%d.h5' % epoch)
    else:
        generator.load_weights('./models/dcgan_generator_epoch_%d.h5' % epoch)
    generatedImages = generator.predict(noise)
    #print generatedImages.shape
    #print generatedImages[1]
    generatedImages = generatedImages/2 + 0.5
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i].transpose((1,2,0)), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    if single:
        plt.savefig('images/single/dcgan_generated_image_SIngle_epoch_%d.png' % epoch)
    else:
        plt.savefig('images/dcgan_generated_image_SIngle_epoch_%d.png' % epoch)


# Create a wall of generated Cifar10 images
def plotDiffGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(16, 16)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generator.load_weights('models/diffGan/diffgan_generator_epoch_%d.h5' % epoch)
    generatedImages = generator.predict(noise)
    #print generatedImages.shape
    #print generatedImages[1]
    generatedImages = generatedImages/2 + 0.5
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i].transpose((1,2,0)), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/diffGan/dcgan_generated_image_epoch_%d.png' % epoch)

if __name__ == '__main__':
    plotDataSetImages()
    #plotGeneratedImages(1)

