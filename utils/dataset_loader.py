import numpy as np
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical


def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_test


def dataset(x):
    if x == 'mnist':
        # Load MNIST data set & reshape them from (n, 28, 28) to (n, 28, 28, 1)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(x_train.shape)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        # normalize input data from [0, 255] to [0, 1]
        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = normalize(x_train, x_test)

        # converting y data into categorical (one-hot encoding)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Load test data and choose a random data out of the test data
        # choose a random data out of the test dataset, if you want to specifically use 'n'th data, x_input = x_test[[n]]
        x_random = np.random.choice(len(x_test), 1)
        x_random_input = x_test[x_random]  # x_test[[7983]]

        return x_train, x_test, y_train, y_test, x_random_input

    elif x == 'cifar10':
        # x_train.shape = (50000, 32, 32 ,3)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # convert the input data from integers to floats and normalize from [0, 255] to [0, 1]
        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = normalize(x_train, x_test)

        # converting y data into categorical (one-hot encoding)
        num_classes = 10
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Load test data and choose a random data out of the test data
        # choose a random data out of the test dataset, if you want to specifically use 'n'th data, x_input = x_test[[n]]
        x_random = np.random.choice(len(x_test), 1)
        x_random_input = x_test[x_random]  # x_test[[7983]]

        return x_train, x_test, y_train, y_test, x_random_input
