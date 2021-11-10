import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import math
import tensorflow as tf
import horovod.keras as hvd

# Initialize Horovod.
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

stack_n = 5
num_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3
batch_size = 128
epochs = 200
iterations = 50000 // batch_size // hvd.size()
weight_decay = 0.0001
mean = [125.307, 122.95, 113.865]
std = [62.9932, 62.0887, 66.7048]


def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 150:
        return 0.01
    return 0.001


def residual_network(img_input, classes_num=10, stack_n=5):
    def residual_block(intput, out_channel, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(
            out_channel,
            kernel_size=(3, 3),
            strides=stride,
            padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(
            out_channel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        if increase:
            projection = Conv2D(
                out_channel,
                kernel_size=(1, 1),
                strides=(2, 2),
                padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

    # build model
    # total layers = stack_n * 3 * 2 + 2
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num,
              activation='softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output = residual_network(img_input, num_classes, stack_n)
    resnet = Model(img_input, output)
    print(resnet.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    sgd = hvd.DistributedOptimizer(sgd)
    resnet.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'])

    cbks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        LearningRateScheduler(scheduler)
    ]
    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        cbks.append(keras.callbacks.TensorBoard('./logs'))

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
