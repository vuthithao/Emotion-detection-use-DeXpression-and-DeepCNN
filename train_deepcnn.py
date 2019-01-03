import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import argparse
from keras.preprocessing.image import img_to_array

from keras.applications.inception_v3 import InceptionV3

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils, to_categorical
import cv2
import os

#------------------------------
#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
#------------------------------

def train(num_class,batch_size ,imagePaths, logdir, epochs):

    def get_model():
        # Initialising the CNN
        model = Sequential()

        # 1 - Convolution
        model.add(Conv2D(64,(3,3), padding='same', input_shape=(224, 224,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2nd Convolution layer
        model.add(Conv2D(128,(5,5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd Convolution layer
        model.add(Conv2D(512,(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th Convolution layer
        model.add(Conv2D(512,(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        # Flattening
        model.add(Flatten())

        # Fully connected layer 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))


        # Fully connected layer 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(num_class, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy'
            , optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        , metrics=['accuracy'])
        return model

    #model.load_weights(weights)
    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        return [mcp_save, reduce_lr_loss, tensorboard]
    
    #gen data
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
            print(label_0)
            index= emotions.index(label_0)
            label_0 = index
            labels.append(label_0)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    # convert the labels from integers to vectors
    #labels = to_categorical(labels, num_classes=num_class)

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
        y_train_cv = to_categorical(y_train_cv, num_classes=num_class)
        y_valid_cv = to_categorical(y_valid_cv, num_classes=num_class)
        name_weights = "final_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
        generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
        model = get_model()
        model.fit_generator(
                generator,
                steps_per_epoch=len(X_train_cv)/batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=1,
                validation_data = (X_valid_cv, y_valid_cv),
                callbacks = callbacks)
    
        print(model.evaluate(X_valid_cv, y_valid_cv))

    model.save_weights('weights_deepcnn_224_kfold.h5')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--imagePaths', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='./logs_deep')
    #parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Starting train...")
    train(args.num_class,args.batch_size, args.imagePaths, args.logdir, args.epochs)
