# import numpy as np
# import keras
# from keras.callbacks import TensorBoard, ModelCheckpoint
from __future__ import print_function
from __future__ import absolute_import
from keras.models import Sequential
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, merge, Flatten, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# import keras.layers
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
# from imagenet_utils import decode_predictions
#from keras.applications.vgg16 import preprocess_input, decode_prediction
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
#from keras.layers import concatenate as concat
from keras import regularizers

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# from imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
# from __future__ import print_function
# from __future__ import absolute_import
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imutils import paths
from sklearn.model_selection import train_test_split, StratifiedKFold

import cv2
import keras.layers
import numpy as np
import keras
import tensorflow as tf


import os, sys
# from imutils import paths
import random
import argparse

#------------------------------
#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
#------------------------------

def train(batch_size, train_dir, validation_dir, logdir, epochs, weight):
    def DeXpression(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                padding=None,
                classes=6):
        # Check weights
        if weights not in {'dexpression', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `dexpression` '
                             '(pre-training on ImageNet).')

        if weights == 'imagenet' and include_top and classes != 6:
            raise ValueError('If using `weights` as dexpression with `include_top`'
                             ' as true, `classes` should be 6')
        # Determine proper input shape
        input_shape = _obtain_input_shape(
            input_shape,
            default_size=224,
            min_size=139,
            data_format=K.image_data_format(),
            require_flatten=include_top)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        # START MODEL
        conv_1 = Convolution2D(64, (7, 7), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(img_input)
        maxpool_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
        x = BatchNormalization()(maxpool_1)

        # FEAT-EX1
        conv_2a = Convolution2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_2a')(x)
        conv_2b = Convolution2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_2b')(conv_2a)
        maxpool_2a = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2a')(x)
        conv_2c = Convolution2D(64, (1, 1), strides=(1,1), name='conv_2c')(maxpool_2a)
        concat_1 = concatenate([conv_2b,conv_2c],axis=3,name='concat_2')
        maxpool_2b = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2b')(concat_1)

        # FEAT-EX2
        conv_3a = Convolution2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_3a')(maxpool_2b)
        conv_3b = Convolution2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_3b')(conv_3a)
        maxpool_3a = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3a')(maxpool_2b)
        conv_3c = Convolution2D(64, (1, 1), strides=(1,1), name='conv_3c')(maxpool_3a)
        concat_3 = concatenate([conv_3b,conv_3c],axis=3,name='concat_3')
        maxpool_3b = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3b')(concat_3)

        # FINAL LAYERS
        net = Flatten()(maxpool_3b)
        net = Dense(classes, activation='softmax',name='predictions')(net)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, net, name='deXpression')
        return model

    def get_callbacks():
        earlyStoping = EarlyStopping(monitor='val_loss', patience=10)
        filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
        mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
#         reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        return [mcp_save, tensorboard]


# initialize the number of epochs to train for, initial learning rate,
# and batch size
    INIT_LR = 1e-3
    train_datagen = ImageDataGenerator(rescale=1./255)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(224, 224),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    callbacks = get_callbacks()
#         generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
    model= DeXpression(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            padding='same',
                            classes=6)
    if weight:
       model.load_weights(weight)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
        metrics=["accuracy"])
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=callbacks)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--validation_dir', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Starting train...")
    train(args.batch_size, args.train_dir, args.validation_dir, args.logdir, args.epochs, args.weight)

