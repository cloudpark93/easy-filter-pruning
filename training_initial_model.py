import matplotlib.pyplot as plt
import os
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

from utils.dataset_loader import dataset
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
#                                            Training Functions                                                        #
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
plt.savefig('vgg16_cifar10_training_accuracy.jpg', dpi=300)
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('vgg16_cifar10_training_loss.jpg', dpi=300)
plt.close()