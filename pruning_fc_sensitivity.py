import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras import backend as K

from utils.dataset_loader import dataset
from utils.pruning_method_fc import pruning_method_fc
from model.model_architectures import model_type

"""    GPU enable and enables running the script without errors    """
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

########################################################################################################################
#                                           Function: FC layer pruning                                                 #
########################################################################################################################

def pruning_filters_fc(pruning_index, layer_to_prune, model_for_pruning, method):

    original_num_filters = [4096, 512]

    pruning_amount = [int(original_num_filters[i] * pruning_index[i]) for i in range(len(original_num_filters))]

    model_pruned = pruning_method_fc(model_for_pruning, layer_to_prune, pruning_amount, method)

    sgd = SGD(lr=1e-3, decay=5e-4, momentum=0.9, nesterov=True)
    model_pruned.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model_pruned

########################################################################################################################
#                   Pruning Fully connected layers to test the sensitivity of each layer to pruning                    #
########################################################################################################################
#
""" Load dataset """
x_train, x_test, y_train, y_test, x_random_input = dataset('cifar10')
method = 'L1norm'

# load the trained initial model to prune
model = load_model('test_model_storage/vgg16_cifar10-450-0.93.h5')

count = -1
layer_to_prune_original_model_fc = []
layer_to_prune_for_continuous_pruning_fc = []

for layer in model.layers:
    count = count + 1
    if 'dense' == (layer.name).split('_')[0]:
        layer_to_prune_original_model_fc.append(count)
        layer_to_prune_for_continuous_pruning_fc.append(count + 1)

# excluding the last dense layer (softmax part)
del layer_to_prune_original_model_fc[-1]
del layer_to_prune_for_continuous_pruning_fc[-1]

pruning_index_per = 0.05 # 0.05 = 5% of the filters are to be pruned
pruning_index_temp = np.ones((len(layer_to_prune_original_model_fc),)) * pruning_index_per

# For pruning job
for layer_to_prune in range(0, len(layer_to_prune_original_model_fc)):
    if os.path.isdir('test_continuous_pruning/fc/test_continuous_pruning_layer{}'.format(layer_to_prune + 1)) == False:
        os.makedirs('test_continuous_pruning/fc/test_continuous_pruning_layer{}'.format(layer_to_prune + 1))

    pruning_index = [pruning_index_temp[layer] if layer == layer_to_prune else 0 for layer in range(len(pruning_index_temp))]
    for i in range(1, int(1/pruning_index_per)):
        if i == 1:
            # load model to prune
            model_pruned = model
            # prune & save conv layer first
            model_pruned = pruning_filters_fc(pruning_index, layer_to_prune_original_model_fc, model_pruned, method)
            model_pruned.save('test_continuous_pruning/fc/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100))

        else:
            # load model to prune
            model_pruned = load_model('test_continuous_pruning/fc/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune + 1, method, pruning_index_per * 100 * (i - 1)))
            # prune & save conv layer
            model_pruned = pruning_filters_fc(pruning_index, layer_to_prune_for_continuous_pruning_fc, model_pruned, method)
            model_pruned.save('test_continuous_pruning/fc/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100*i))

        del model_pruned
        K.clear_session()

        # Evaluation after pruning
        model_pruned = load_model('test_continuous_pruning/fc/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100*i))
        results = model_pruned.evaluate(x_test, y_test, verbose=0)
        np.savetxt('test_continuous_pruning/fc/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.csv'.format(layer_to_prune+1, method, pruning_index_per*100*i), results, delimiter=',')
        print('Test loss for pruned model: ', results[0])
        print('Test accuracy for pruned model: ', results[1])

        del model_pruned
        K.clear_session()



