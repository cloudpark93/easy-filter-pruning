import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from matplotlib import colors as mcolors


from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras import backend as K

from utils.dataset_loader import dataset
from utils.pruning_method_conv import pruning_method_conv
from utils.pruning_method_fc import pruning_method_fc

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
#                                     Function: Conv & FC layer pruning                                                #
########################################################################################################################


def pruning_filters_conv(pruning_index, layer_to_prune, model_for_pruning, method):

    original_num_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    pruning_amount = [int(original_num_filters[i] * pruning_index[i]) for i in range(len(original_num_filters))]

    model_pruned = pruning_method_conv(model_for_pruning, layer_to_prune, pruning_amount, method)

    sgd = SGD(lr=1e-3, decay=5e-4, momentum=0.9, nesterov=True)
    model_pruned.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model_pruned


def pruning_filters_fc(pruning_index, layer_to_prune, model_for_pruning, method):

    original_num_filters = [4096, 512]

    pruning_amount = [int(original_num_filters[i] * pruning_index[i]) for i in range(len(original_num_filters))]

    model_pruned = pruning_method_fc(model_for_pruning, layer_to_prune, pruning_amount, method)

    sgd = SGD(lr=1e-3, decay=5e-4, momentum=0.9, nesterov=True)
    model_pruned.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model_pruned

########################################################################################################################
#                    Pruning Convolutional layers to test the sensitivity of each layer to pruning                     #
########################################################################################################################


""" Load dataset """
x_train, x_test, y_train, y_test, x_random_input = dataset('cifar10')
method = 'L1norm'

# load the trained initial model to prune
model = load_model('test_model_storage/vgg16_cifar10-450-0.93.h5')

count = -1
layer_to_prune_original_model_conv = []
layer_to_prune_for_continuous_pruning_conv = []

for layer in model.layers:
    count = count + 1
    if 'conv2d' == (layer.name).split('_')[0]:
        layer_to_prune_original_model_conv.append(count)
        layer_to_prune_for_continuous_pruning_conv.append(count+1)


pruning_index_per = 0.1 # 0.05 = 5% of the filters are to be pruned
pruning_index_temp = np.ones((len(layer_to_prune_original_model_conv),)) * pruning_index_per

# For pruning job
for layer_to_prune in range(0, len(layer_to_prune_original_model_conv)):
    if os.path.isdir('test_continuous_pruning/conv/test_continuous_pruning_layer{}'.format(layer_to_prune + 1)) == False:
        os.makedirs('test_continuous_pruning/conv/test_continuous_pruning_layer{}'.format(layer_to_prune + 1))

    pruning_index = [pruning_index_temp[layer] if layer == layer_to_prune else 0 for layer in range(len(pruning_index_temp))]
    for i in range(1, int(1/pruning_index_per)):
        if i == 1:
            # load model to prune
            model_pruned = model
            # prune & save conv layer first
            model_pruned = pruning_filters_conv(pruning_index, layer_to_prune_original_model_conv, model_pruned, method)
            model_pruned.save('test_continuous_pruning/conv/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100))

        else:
            # load model to prune
            model_pruned = load_model('test_continuous_pruning/conv/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune + 1, method, pruning_index_per * 100 * (i - 1)))
            # prune & save conv layer
            model_pruned = pruning_filters_conv(pruning_index, layer_to_prune_for_continuous_pruning_conv, model_pruned, method)
            model_pruned.save('test_continuous_pruning/conv/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100*i))

        del model_pruned
        K.clear_session()

        # Evaluation after pruning
        model_pruned = load_model('test_continuous_pruning/conv/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100*i))
        results = model_pruned.evaluate(x_test, y_test, verbose=0)
        np.savetxt('test_continuous_pruning/conv/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.csv'.format(layer_to_prune+1, method, pruning_index_per*100*i), results, delimiter=',')
        print('Test loss for pruned model: ', results[0])
        print('Test accuracy for pruned model: ', results[1])

        del model_pruned
        K.clear_session()

########################################################################################################################
#                   Pruning Fully connected layers to test the sensitivity of each layer to pruning                    #
########################################################################################################################

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

pruning_index_per = 0.1 # 0.05 = 5% of the filters are to be pruned
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

########################################################################################################################
#                                   Plotting sensitivity of Conv layer to pruning                                      #
########################################################################################################################

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
pruning_index = 10.0
plot_line_color = ["r", "g", "b", "k", "y", "m", "c"]
plot_line_style = ["-", "--"]

file_sub_name = ["L1norm"]
pruning_method = ["L1-norm"]

num_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

for method_count in range(1):
    conv_layers_acc = []
    conv_layers_loss = []
    file_number = sum([len(d) for r, d, files in os.walk("test_continuous_pruning/conv/")])

    for layer in range(1, file_number+1):
        pruned_loss = []
        pruned_acc = []

        for i in range(1, int(1/(pruning_index/100))):
            data = csv.DictReader(open("test_continuous_pruning/conv/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.csv"
                                       .format(layer, file_sub_name[method_count], pruning_index * i)))
            for raw in data:
                pruned_acc.append(list(raw.values()))
                pruned_loss.append(list(raw.keys()))

        if layer == 1:
            conv_layers_acc = np.array(pruned_acc).astype(np.float32)
            conv_layers_loss = np.array(pruned_loss).astype(np.float32)
        else:
            conv_layers_acc = np.append(conv_layers_acc, np.array(pruned_acc).astype(np.float32), axis=1)
            conv_layers_loss = np.append(conv_layers_loss, np.array(pruned_loss).astype(np.float32), axis=1)

    plt.style.use("ggplot")
    plt.figure("Acc figure {} method Conv layer{}".format(file_sub_name[method_count], layer))
    x = range(int(pruning_index), 100, int(pruning_index))
    for layer in range(0, file_number):
        plt.plot(x, conv_layers_acc[:, layer], linestyle=plot_line_style[layer // len(plot_line_color)],
                 marker='o', color=plot_line_color[layer % len(plot_line_color)], label="conv_{} {}".format(layer+1, num_filters[layer]),
                 linewidth=1.0, markersize=2)

    plt.ylim(0.0, 1.0)
    plt.title("CIFAR10 VGG-16 Pruning Accuracy \n{} method".format(pruning_method[method_count]))
    plt.xlabel("Filters Pruned Away (%)")
    plt.ylabel("Accuracy")
    plt.legend(loc=3)
    plt.savefig('test_continuous_pruning/CIFAR10 VGG-16 Pruning Conv Layer Accuracy {} method.jpg'.format(pruning_method[method_count]), dpi=300)
    print('Conv layer accuracy image saved successfully')
    plt.close()

########################################################################################################################
#                                    Plotting sensitivity of Fc layer to pruning                                       #
########################################################################################################################

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
pruning_index = 10.0
plot_line_color = ["r", "g", "b", "k", "y", "m", "c"]
plot_line_style = ["-", "--"]

file_sub_name = ["L1norm"]
pruning_method = ["L1-norm"]

num_filters = [4096, 512]

for method_count in range(1):
    fc_layers_acc = []
    fc_layers_loss = []
    file_number = sum([len(d) for r, d, files in os.walk("test_continuous_pruning/fc/")])

    for layer in range(1, file_number+1):
        pruned_loss = []
        pruned_acc = []

        for i in range(1, int(1/(pruning_index/100))):
            data = csv.DictReader(open("test_continuous_pruning/fc/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.csv"
                                       .format(layer, file_sub_name[method_count], pruning_index * i)))
            for raw in data:
                pruned_acc.append(list(raw.values()))
                pruned_loss.append(list(raw.keys()))

        if layer == 1:
            fc_layers_acc = np.array(pruned_acc).astype(np.float32)
            fc_layers_loss = np.array(pruned_loss).astype(np.float32)
        else:
            fc_layers_acc = np.append(fc_layers_acc, np.array(pruned_acc).astype(np.float32), axis=1)
            fc_layers_loss = np.append(fc_layers_loss, np.array(pruned_loss).astype(np.float32), axis=1)

    plt.style.use("ggplot")
    plt.figure("Acc figure {} method Fc layer{}".format(file_sub_name[method_count], layer))
    x = range(int(pruning_index), 100, int(pruning_index))
    for layer in range(0, file_number):
        plt.plot(x, fc_layers_acc[:, layer], linestyle=plot_line_style[layer // len(plot_line_color)],
                 marker='o', color=plot_line_color[layer % len(plot_line_color)], label="conv_{} {}".format(layer+1, num_filters[layer]),
                 linewidth=1.0, markersize=2)

    plt.ylim(0.0, 1.0)
    plt.title("CIFAR10 VGG-16 Pruning Accuracy \n{} method".format(pruning_method[method_count]))
    plt.xlabel("Filters Pruned Away (%)")
    plt.ylabel("Accuracy")
    plt.legend(loc=3)
    plt.savefig('test_continuous_pruning/CIFAR10 VGG-16 Pruning Fc Layer Accuracy {} method.jpg'.format(pruning_method[method_count]), dpi=300)
    print('FC layer accuracy image saved successfully')
    plt.close()