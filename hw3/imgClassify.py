# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 13:26:56 2016

@author: DennisShen
@title : image classification
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import Input, UpSampling2D
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend as K

import cPickle as pickle
import numpy as np
import sys

def load_data(label_file, unlabel_file, test_file):
    label_data   = pickle.load(open(label_file, 'rb'))
    unlabel_data = pickle.load(open(unlabel_file, 'rb'))
    test_data    = pickle.load(open(test_file, 'rb'))
    
    return label_data, unlabel_data, test_data
    
def process_data(label_data, unlabel_data, test_data, use_test_to_train='F'):
    label_data_array   = np.asarray(label_data)
    unlabel_data_array = np.asarray(unlabel_data)
    test_data_array    = np.asarray(test_data['data'])

    X_label   = np.resize(label_data_array, (5000, 3, 32, 32))
    X_unlabel = np.resize(unlabel_data_array, (45000, 3, 32, 32))
    X_test    = np.resize(test_data_array, (10000, 3, 32, 32))
    
    X_label   = X_label.astype('float32')
    X_unlabel = X_unlabel.astype('float32')
    X_test    = X_test.astype('float32')

    X_label   /= 255
    X_unlabel /= 255
    X_test    /= 255
    
    if use_test_to_train == 'T':
        X_unlabel = np.concatenate((X_unlabel, X_test), axis=0)        

    return X_label, X_unlabel, X_test
    
def split_validation(X):
    y = np.repeat(np.array(range(10)), 500)
    
    random_indexes = np.random.permutation(len(y))
    train_inds = random_indexes[:int((0.9*len(y)))]
    valid_inds = random_indexes[int((0.9*len(y))):]
    return X[train_inds], y[train_inds], X[valid_inds], y[valid_inds]

def do_CNN(X_train_label, y_train_label, X_valid_label, y_valid_label, X_unlabel, self_learning, clustering, y_unlabel, model_name):
    batch_size        = 32
    nb_classes        = 10
    nb_epoch          = 200
    data_augmentation = True
    threshold         = 0.9
    iteration         = 2 if self_learning == 'T' else 1
    
    if clustering == 'T':
        X_train_label = np.concatenate((X_train_label, X_unlabel), axis=0)
        y_train_label = np.concatenate((y_train_label, y_unlabel), axis=0)
    
    print('X_train shape:', X_train_label.shape)
    print(X_train_label.shape[0], 'train samples')
    print(X_valid_label.shape[0], 'valid samples')
    
    Y_train_label = np_utils.to_categorical(y_train_label, nb_classes)
    Y_valid_label = np_utils.to_categorical(y_valid_label, nb_classes)
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train_label.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    X_train_new = X_train_label
    Y_train_new = Y_train_label
    
    if self_learning == 'T':
        print('Use semi-supervise learning...\n')
        
    for i in range(iteration):
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(X_train_new, Y_train_new,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_data=(X_valid_label, Y_valid_label),
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                                         featurewise_center=False,            # set input mean to 0 over the dataset
                                         samplewise_center=False,             # set each sample mean to 0
                                         featurewise_std_normalization=False, # divide inputs by std of the dataset
                                         samplewise_std_normalization=False,  # divide each input by its std
                                         zca_whitening=False,                 # apply ZCA whitening
                                         rotation_range=0,                    # randomly rotate images in the range (degrees, 0 to 180)
                                         width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
                                         height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
                                         horizontal_flip=True,                # randomly flip images
                                         vertical_flip=False)                 # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(X_train_new)

            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(X_train_new, Y_train_new,
                                             batch_size=batch_size),
                                             samples_per_epoch=X_train_new.shape[0],
                                             nb_epoch=nb_epoch,
                                             validation_data=(X_valid_label, Y_valid_label))
       
        if self_learning == 'T':
            unlabel_predict_prob  = model.predict_proba(X_unlabel)
            unlabel_predict_class = model.predict_classes(X_unlabel)
    
            X_train_add = X_unlabel[np.amax(unlabel_predict_prob, axis=1) >= threshold]
            X_train_new = np.concatenate((X_train_label, X_train_add), axis=0)
        
            Y_train_temp = np.array([], dtype=np.int64)
            Y_train_temp = np.append(Y_train_temp, unlabel_predict_class[np.amax(unlabel_predict_prob, axis=1) >= threshold], axis=0)
            Y_train_temp = np.resize(Y_train_temp, (Y_train_temp.shape[0], 1))
            Y_train_add  = np_utils.to_categorical(Y_train_temp, nb_classes)
            Y_train_new  = np.concatenate((Y_train_label, Y_train_add), axis=0)
    
    print('save model...\n')
    model.save(model_name+'.h5')

def img_clustering(X_label, X_unlabel, X_test): 
    input_img = Input(shape=(3, 32, 32))

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    encoded = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)

    # at this point the representation is (256, 1, 1) i.e. 256-dimensional

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    
    X_all = np.concatenate((X_label, X_unlabel), axis=0)
    
    autoencoder.fit(X_all, X_all,
                    nb_epoch=20,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_test, X_test))
    
    encoder = Model(input_img, encoded)
    
    X_label_code   = encoder.predict(X_label)
    X_unlabel_code = encoder.predict(X_unlabel)
    
    X_label_code   = np.reshape(X_label_code, (5000, 256))
    X_unlabel_code = np.reshape(X_unlabel_code, (45000, 256))
    
    cos_sim = cosine_similarity(X_unlabel_code, X_label_code)
    cos_sim = np.reshape(cos_sim, (45000, 10, 500))
    cos_sim = np.average(cos_sim, axis=2)    
    y_unlabel = np.argmax(cos_sim, axis=1)
    
    return y_unlabel
    
def do_test(predict, model_file, X_test):
    model = load_model(model_file+'.h5')
    
    test_predict_class = model.predict_classes(X_test)
    
    with open(predict, 'w') as f:
        f.write('ID,class\n')
        for idx in range(10000):
            f.write('%d,%d\n'%(idx, test_predict_class[idx]))
    
def run_img_classification(is_train, path, model, predict='', self_learning='F', clustering='F', use_test_to_train='F'):
    print 'load data...\n'
    label_data, unlabel_data, test_data = load_data(path + 'all_label.p',
                                                    path + 'all_unlabel.p',
                                                    path + 'test.p')
    
    print 'process data...\n'
    X_label, X_unlabel, X_test = process_data(label_data, unlabel_data, test_data, use_test_to_train)
    
    if is_train=='train':
        y_unlabel = []
        if clustering == 'T':
            print('image clustering...')
            y_unlabel = img_clustering(X_label, X_unlabel, X_test)
    
        print 'split data...\n'
        X_train_label, y_train_label, X_valid_label, y_valid_label = split_validation(X_label)
    
        print 'CNN...\n'
        do_CNN(X_train_label, y_train_label, X_valid_label, y_valid_label, X_unlabel, 
               self_learning, clustering, y_unlabel, model)
    else:
        print 'predict testing data...\n'
        do_test(predict, model, X_test)

if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    if sys.argv[1]=='train':
        run_img_classification(is_train=sys.argv[1], path=sys.argv[2], model=sys.argv[3], 
                               self_learning=sys.argv[4], clustering=sys.argv[5], use_test_to_train=sys.argv[6])
    else:
        run_img_classification(is_train=sys.argv[1], path=sys.argv[2], model=sys.argv[3], predict=sys.argv[4])
        
        
