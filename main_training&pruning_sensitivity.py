import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras import backend as K

from utils.dataset_loader import dataset
from utils.pruning_method_conv import pruning_method_conv
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
#                                            Functions BEGIN                                                           #
########################################################################################################################

""" Set a model save directory """

model_save_folder_path = './test_model_storage/'
if not os.path.exists(model_save_folder_path):
    os.mkdir(model_save_folder_path)


""" Training a model """

def training_model_with_graph(model, x_train, x_test, y_train, y_test, epochs):
    batch_size = 64
    learning_rate = 0.01 # need to set the same value as the 'lr in model_architecture'
    lr_drop = 20
    period = 50 # saving model in every period, eg) 50 = save model in every 50 epochs

    tensorboard = TensorBoard(log_dir='./tensorboard')

    def lr_scheduler(epoch):
        return learning_rate * (tf.math.exp(-0.1) ** (epoch//lr_drop))
    reduce_lr = LearningRateScheduler(lr_scheduler)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(x_train)

    filename = 'vgg16_cifar10-{epoch:02d}-{val_accuracy:.2f}.h5'
    model_path = model_save_folder_path + filename
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_weights_only=False, period=period)

    history = model.fit_generator(datagen.flow(x=x_train, y=y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0]//batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  callbacks=[reduce_lr, tensorboard, cb_checkpoint])

    return model, history


""" Pruning """
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
#                                            Initial model training                                                    #
########################################################################################################################

""" Load dataset """
x_train, x_test, y_train, y_test, x_random_input = dataset('cifar10')
print('dataset loaded successfully')

""" Define an initial model architecture to train """
initial_model = model_type('vgg16')
initial_model.summary()
""" Train and save a model (model architecture & weight together) """
model_before_prune, history = training_model_with_graph(initial_model, x_train, x_test, y_train, y_test, epochs=450)

# convert the history.history dictionary to a pandas DataFrame and save it as csv
history_df = pd.DataFrame(history.history)
history_df_csv = model_save_folder_path + 'history.csv'
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
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


########################################################################################################################
#                                         Pruning Convolutional layers                                                 #
########################################################################################################################
#
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

pruning_index_per = 0.05 # 0.05 = 5% of the filters are to be pruned
pruning_index_temp = np.ones((len(layer_to_prune_original_model_conv),)) * pruning_index_per 
pruning_index = np.ones((len(layer_to_prune_original_model_conv),)) * pruning_index_per

# For pruning job
for layer_to_prune in range(0, len(layer_to_prune_original_model_conv)):
    if os.path.isdir('test_continuous_pruning/test_continuous_pruning_layer{}'.format(layer_to_prune + 1)) == False:
        os.makedirs('test_continuous_pruning/test_continuous_pruning_layer{}'.format(layer_to_prune + 1))

    pruning_index = [pruning_index_temp[layer] if layer == layer_to_prune else 0 for layer in range(len(pruning_index_temp))]
    for i in range(1, int(1/pruning_index_per)):
        if i == 1:
            # load model to prune
            model_pruned = load_model('test_model_storage/vgg16_cifar10-450-0.93.h5')
            # prune & save conv layer first
            model_pruned = pruning_filters_conv(pruning_index, layer_to_prune_original_model_conv, model_pruned, method)
            model_pruned.save('test_continuous_pruning/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1, method, pruning_index_per*100))

        else:
            # load model to prune
            model_pruned = load_model('test_continuous_pruning/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune + 1, method, pruning_index_per * 100 * (i - 1)))
            # prune & save conv layer
            model_pruned = pruning_filters_conv(pruning_index, layer_to_prune_for_continuous_pruning_conv, model_pruned, method)
            model_pruned.save('test_continuous_pruning/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1,method, pruning_index_per*100*i))

        # Evaluation after pruning
        model_pruned = load_model('test_continuous_pruning/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.h5'.format(layer_to_prune+1,method, pruning_index_per*100*i))
        results = model_pruned.evaluate(x_test, y_test, verbose=0)
        np.savetxt('test_continuous_pruning/test_continuous_pruning_layer{}/vgg16_cifar10_after_prune_{}_{}%.csv'.format(layer_to_prune+1,method, pruning_index_per*100*i), results, delimiter=',')
        print('Test loss for pruned model: ', results[0])
        print('Test accuracy for pruned model: ', results[1])

        del model_pruned
        K.clear_session()
        print('deleted')



