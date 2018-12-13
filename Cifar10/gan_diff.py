'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse
from scipy.misc import imsave


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten,Lambda,Permute
from keras.layers import UpSampling2D,Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import cifar10
from keras import initializers
from keras.optimizers import Adam

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
import  random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from show_cifar10 import plotDiffGeneratedImages
from util import *
random.seed(100)





# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
# parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])#
parser.add_argument('weight_diff',default=0.5, help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', default = 1, help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', default= 0.001, help="step size of gradient descent", type=float)
parser.add_argument('threshold', default= 0,help="threshold for determining neuron activated", type=float)
parser.add_argument('digit',default = 1, help="digits to retrieve", type=int)
parser.add_argument('ratio', default=5, help="params for diff/gan ratios in iterations", type=int)
args = parser.parse_args()


K.set_image_dim_ordering('th')



img_rows, img_cols = 32, 32
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#get all the ones data
ones_train = (x_train.astype(np.float32) - 127.5)/127.5
ones_train  = ones_train[np.where(y_train == args.digit)[0]]



# load multiple models sharing same input tensor
# model1 = Model1(input_tensor=input_tensor)
# model2 = Model2(input_tensor=input_tensor)
# model3 = Model3(input_tensor=input_tensor)

# # init coverage table

# model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)


# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim  = 100
adam = Adam(lr=0.0002, beta_1=0.5)


# Generator
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

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape= ones_train.shape[1:], kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

#print(x.shape)
# in deepexplore code       input_shape = (img_rows, img_cols, 1) tf
# in gan code               input_shape = (1,img_rows, img_cols) th

generator.load_weights('./models/singleModel/dcgan_generator_epoch_%d.h5'% 100)
print(bcolors.OKBLUE + 'Model params generator loaded' + bcolors.ENDC)

discriminator.load_weights('./models/singleModel/dcgan_discriminator_epoch_%d.h5'% 100)
print(bcolors.OKBLUE + 'Model params discriminator loaded' + bcolors.ENDC)

ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)
#print(gan.summary())
#
# gen_img = generator.predict()
#
# # orig_img = gen_img.copy()


adam2 = Adam(lr = args.step, beta_1=0.5)
#actually we don't care all output of model1,2,3 but only the category that we focus on
#model = Model( x)

K.set_image_dim_ordering('tf')
#input_shape = (img_rows, img_cols, 3)
input_shape = (32, 32, 3)
input_tensor = Input(shape=input_shape)
print(input_tensor.shape)
model1 = Model1(input_tensor= input_tensor)
model2 = Model2(input_tensor= input_tensor)
model3 = Model3(input_tensor= input_tensor)

model1.trainable = False
model2.trainable = False
model3.trainable = False

orig_label = 1


print(x.shape)
x = Permute((2,3,1))(x)
m1output = model1(x)# * args.weight_diff
#neu1_output = model1.get_layer("block2_pool1").output

m2output = model2(x)
m3output = model3(x)

def sle(x):
    return tf.gather(x, 1, axis =1)

o1 = Lambda(sle,name = "oo1")(m1output)
o2 = Lambda(sle,name = "oo2")(m2output)
o3 = Lambda(sle,name = "oo3")(m3output)

o11 = Reshape((1,), name = "o1")(o1)
o22 = Reshape((1,), name = "o2")(o1)
o33 = Reshape((1,), name = "o3")(o1)
print(o1.shape)
DiffNetwork = Model(outputs = [o11, o22, o33], input = ganInput)
DiffNetwork.compile(loss = {"o1":"binary_crossentropy",
                            "o2":"binary_crossentropy",
                            "o3":"binary_crossentropy",
                            },
                    optimizer = adam2,
                    loss_weights={"o1":args.weight_diff, "o2":0.01,"o3":0.01})
print(DiffNetwork.summary())

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/diffGan/diffgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/diffGan/diffgan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = x_test.shape[0] / batchSize
    #batchCount = 25
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in xrange(1, epochs+1):
        diffEpoch = 0
        print('-'*15, 'Epoch %d' % e, '-'*15)
        saveModels(0)
        plotDiffGeneratedImages(0)
        #batchCount = 10
        for j in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = ones_train[np.random.randint(0, ones_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2 * batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

            label1 = np.argmax(model1.predict(generatedImages.transpose((0, 3, 2, 1))), axis=1)
            label2 = np.argmax(model2.predict(generatedImages.transpose((0, 3, 2, 1))), axis=1)
            label3 = np.argmax(model3.predict(generatedImages.transpose((0, 3, 2, 1))), axis=1)

            labeltotrain = (np.abs(label1 - label2) + np.abs(label2 - label3))!=0
            diffCount = labeltotrain.sum()
            diffEpoch += diffCount
            diffTrain = np.repeat(labeltotrain, randomDim).reshape((batchSize,randomDim))
            diffTrain = np.extract(diffTrain, noise).reshape((diffCount, randomDim))
            #print(diffTrain.shape)
            if diffCount == 0 or j %args.ratio != 0:
                continue
            diffLoss = DiffNetwork.train_on_batch(diffTrain, {"o1":np.ones(diffCount),"o2":np.zeros(diffCount),"o3":np.zeros(diffCount)})
            print(diffLoss)
            print("Diffrate for this epoch: %d / %d" % (diffCount, batchSize))
            if j % 10 == 0:
                saveModels(j)
                plotDiffGeneratedImages(j)
        print("Diffrate for this epoch: " +str(diffEpoch)+"/"+str(batchCount*batchSize))

        # for adversarial image generation

        # Store loss of most recent batch from this epoch
        # dLosses.append(dloss)
        # gLosses.append(gloss)
        if e < 5 or e % 5 == 0:
            saveModels(e)
            plotDiffGeneratedImages(e)
        #saveModels(e)

    # Plot losses from every epoch
    # plotLoss(e)


if __name__ == '__main__':
    train(30, 128)

