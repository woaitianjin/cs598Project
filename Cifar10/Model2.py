'''
LeNet-1
'''

# usage: python MNISTModel1.py - train the model

from __future__ import print_function

from keras.datasets import cifar10
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten,Dropout
from keras.models import Model
from keras.utils import to_categorical

from configs import bcolors


def Model2(input_tensor=None, train=False):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (3, 3)

    if train:
        batch_size = 256
        nb_epoch = 10

        # input image dimensions
        img_rows, img_cols = 32, 32

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print(x_train.shape)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()

    # block1
    x = Convolution2D(32, kernel_size, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(32, kernel_size)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Model2.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Model2.h5')
        print(bcolors.OKBLUE + 'Model2 loaded' + bcolors.ENDC)

    return model


if __name__ == '__main__':
    Model2(train=True)
