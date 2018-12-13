from Model1 import  Model1
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten,Dropout
from keras.models import Model


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)
    model1 = Model1(input_tensor=input_tensor)
# Model1(train=False)