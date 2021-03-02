import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import SGD, Adam
import pandas as pd
import os

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from utils.dataset_loader import dataset
from utils.pruning_method_conv import pruning_method_conv
from utils.pruning_method_fc import pruning_method_fc
from utils.pruning_amount_extraction_based_on_sensitivity import conv_pruning_amount_calculator, fc_pruning_amount_calculator


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
#                                            Functions Training                                                        #
########################################################################################################################

""" Training a model """
def training_model(model, x_train, x_test, y_train, y_test, epochs):
    batch_size = 64

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(x_train)

    history = model.fit_generator(datagen.flow(x=x_train, y=y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0]//batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(x_test, y_test))

    return model, history


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
#                                              Functions END                                                           #
########################################################################################################################

""" Load dataset """
x_train, x_test, y_train, y_test, x_random_input = dataset('cifar10')

method = 'L1norm'
pruning_index_for_sensitivity = 10.0 # sensitivity calculated for every 10% (same value as pruning_sensitivity_calculation&plotting.py script)
pruning_acc_threshold = 0.9 # pruning each layer of accuracy 90% and above

# determine how much you want to prune in each layer based on the pruning_acc_threshold
pruning_index_conv = conv_pruning_amount_calculator(method, pruning_index_for_sensitivity, pruning_acc_threshold)
pruning_index_fc = fc_pruning_amount_calculator(method, pruning_index_for_sensitivity, pruning_acc_threshold)

# load model to prune
model = load_model('test_model_storage/vgg16_cifar10-450-0.93.h5')

count = -1
layer_to_prune_original_model_conv = []
layer_to_prune_for_continuous_pruning_conv = []

for layer in model.layers:
    count = count + 1
    if 'conv2d' == (layer.name).split('_')[0]:
        layer_to_prune_original_model_conv.append(count)
        layer_to_prune_for_continuous_pruning_conv.append(count+1)

layer_to_prune_original_model_fc = []
layer_to_prune_for_continuous_pruning_fc = []

count = -1
for layer in model.layers:
    count = count +1
    if 'dense' == (layer.name).split('_')[0]:
        layer_to_prune_original_model_fc.append(count)
        layer_to_prune_for_continuous_pruning_fc.append(count + 1)

# excluding the last dense layer (softmax part)
del layer_to_prune_original_model_fc[-1]
del layer_to_prune_for_continuous_pruning_fc[-1]

# conv layer pruning
print('conv layer pruning begins')
model_pruned = pruning_filters_conv(pruning_index_conv, layer_to_prune_original_model_conv, model, method)
if os.path.isdir('pruned_model') == False:
    os.makedirs('pruned_model')

model_pruned.save('pruned_model/vgg16_cifar10_pruned_{}.h5'.format(method))
del model_pruned
K.clear_session()

# fc layer pruning
print('fc layer pruning begins')
model_pruned = load_model('pruned_model/vgg16_cifar10_pruned_{}.h5'.format(method))
model_pruned = pruning_filters_fc(pruning_index_fc, layer_to_prune_for_continuous_pruning_fc, model_pruned, method)
model_pruned.save('pruned_model/vgg16_cifar10_pruned_{}.h5'.format(method))

del model_pruned
K.clear_session()

model_pruned = load_model('pruned_model/vgg16_cifar10_pruned_{}.h5'.format(method))
results = model_pruned.evaluate(x_test, y_test, verbose=0)
np.savetxt('pruned_model/vgg16_cifar10_pruned_{}.csv'.format(method), results, delimiter=',')
print('Test loss for pruned model: ', results[0])
print('Test accuracy for pruned model: ', results[1])

epochs = 40
pruned_model_retraining, history = training_model(model_pruned, x_train, x_test, y_train, y_test, epochs)

# convert the history.history dictionary to a pandas DataFrame and save it as csv
history_df = pd.DataFrame(history.history)
save_path = 'pruned_model'
history_df_csv = os.path.join(save_path, 'history_{}.csv'.format(method))
with open(history_df_csv, mode='w') as f:
    history_df.to_csv(f)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('pruned_model/vgg16_cifar10_pruned&retrained_{}_acc.jpg'.format(method), dpi=300)
plt.close()
# plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('pruned_model/vgg16_cifar10_pruned&retrained_{}_loss.jpg'.format(method), dpi=300)
plt.close()
# plt.show()

pruned_model_retraining.save('pruned_model/vgg16_cifar10_pruned&retrained_{}.h5'.format(method))
results = pruned_model_retraining.evaluate(x_test, y_test, verbose=0)
np.savetxt('pruned_model/vgg16_cifar10_pruned&retrained_{}.csv'.format(method), results, delimiter=',')
print('Test loss for pruned model: ', results[0])
print('Test accuracy for pruned model: ', results[1])