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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

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

def train(batch_size, imagePaths, logdir, epochs):
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
        conv_3c = Convolution2D(64, (1, 1), strides=(1,1), name='conv_3c')(maxpool_2a)
        concat_3 = concatenate([conv_3b,conv_3c],axis=3,name='concat_3')
        maxpool_3b = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3b')(concat_3)

        # FINAL LAYERS
        net = Flatten()(maxpool_3b) 
        net = Dense(classes, activation='softmax',kernel_regularizer=regularizers.l2(0.01), name='predictions')(net)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, net, name='deXpression')
        return model

    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
        earlyStoping = EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        return [mcp_save, tensorboard]


# initialize the number of epochs to train for, initial learning rate,
# and batch size
    INIT_LR = 1e-3

    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []
    
    emotions = ('angry','fear', 'happy', 'sad', 'surprise', 'neutral')

    name_paths=os.listdir(imagePaths)

    for i in range (len(name_paths)):
        dirs=os.path.join(imagePaths, name_paths[i])
        name_dirs= os.listdir(dirs)
        for j in range (len (name_dirs)):
            data_dirs=os.path.join(dirs, name_dirs[j])
            print (data_dirs)
            img = cv2.imread(data_dirs)        
            img = cv2.resize(img,(224,224))
            img = img_to_array(img)
            data.append(img)

        # extract the class label from the image path and update the
        # labels list
            label_0 = data_dirs.split(os.path.sep)[5]
            index= emotions.index(label_0)
            label_0 = index
            labels.append(label_0)

 # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
        # k-fold define
    def load_data_kfold(k,trainX, trainY):
        folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(trainX, trainY))
        return folds, trainX, trainY

    k = 10
    folds, X_train, y_train = load_data_kfold(k,data,labels)
    print(X_train.shape)
    print(y_train.shape)
    gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 10
                        )
    for j, (train_idx, val_idx) in enumerate(folds):
    
        print('\nFold ',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid_cv = X_train[val_idx]
        y_valid_cv= y_train[val_idx]
        y_train_cv = to_categorical(y_train_cv, num_classes=6)
        y_valid_cv = to_categorical(y_valid_cv, num_classes=6)
        name_weights = "gen_data_final_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights = name_weights,  patience_lr=10)
        generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
        model= DeXpression(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            padding='same',
                            classes=6)
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.001/epochs)
        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
        
        model.fit_generator(
                generator,
                steps_per_epoch=len(X_train_cv)/batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                validation_data = (X_valid_cv, y_valid_cv),
                callbacks = callbacks)
#         hist= model.fit(X_train_cv, y_train_cv,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(X_valid_cv, y_valid_cv),
#                     callbacks=callbacks)
    
        print(model.evaluate(X_valid_cv, y_valid_cv))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--imagePaths', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='./logs')
    #parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Starting train...")
    train(args.batch_size, args.imagePaths, args.logdir, args.epochs)

