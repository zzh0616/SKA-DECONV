#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import astropy.io.fits as fits
from tensorflow.keras import layers, models, optimizers
import gc
import configparser

from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split

def load_data(name):
    if name == 'None':
        return np.array([])
    data = fits.open(name)[0].data
    return data

def slice_images(images, slice_size=128, stride=128,padding_value=0):
    window_shape = (slice_size, slice_size)
    step = (stride, stride)

    pad_x = 0 if (images.shape[1] - slice_size) % stride == 0 else stride - ((images.shape[1] - slice_size) % stride)
    pad_y = 0 if (images.shape[2] - slice_size) % stride == 0 else stride - ((images.shape[2] - slice_size) % stride)

    pad_width = ((0, 0), (0, pad_x), (0, pad_y))
    images = np.pad(images, pad_width=pad_width, mode='constant', constant_values=padding_value)
    sliced_image = []
    for i in range(images.shape[0]):
        slice_temp = view_as_windows(images[i], window_shape, step)
        slice_temp = slice_temp.reshape((-1, slice_size, slice_size))
        sliced_image.extend(slice_temp)
    return np.array(sliced_image),pad_x,pad_y

def normalize_images(images,mean=None,std=None):
    if mean is None or std is None:
        mean = np.mean(images)
        std = np.std(images)
        return mean, std, (images - mean) / std
    else:
        return (images - mean) / std

def preprocess_images(dirty, true_sky, freq_array, slice_size=128,stride=128):
    mean, std, dirty = normalize_images(dirty)
    xxx,yyy, true_sky = normalize_images(true_sky)

    dirty_sliced, pad_x, pad_y = slice_images(dirty, slice_size, stride)
    true_sky_sliced, pad_x, pad_y = slice_images(true_sky, slice_size, stride)

    num_slices_per_freq = ((dirty.shape[1] - slice_size) // stride + 2)* ((dirty.shape[2] - slice_size) // stride + 2)
    del dirty, true_sky
    gc.collect()
    freq_ind = np.repeat(np.arange(len(freq_array)), num_slices_per_freq)


    X_train,X_temp, freq_train, freq_temp, y_train, y_temp = train_test_split(dirty_sliced, freq_ind,  true_sky_sliced,  test_size=0.2, random_state=42)

    del dirty_sliced, true_sky_sliced
    gc.collect()

    X_val, X_test, freq_val, freq_test, y_val, y_test = train_test_split(X_temp, freq_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, freq_train, freq_val, freq_test, y_train, y_val, y_test

def cdae_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    padding = 'same'
    activate = 'elu'
    x = layers.Conv2D(64, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(inputs)
    x = layers.Conv2D(64, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.Conv2D(128, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.Conv2D(256, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2DTranspose(256, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(encoded)
    x = layers.Conv2DTranspose(256, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=activate, padding=padding, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2DTranspose(1, (3, 3), activation='tanh', padding=padding, kernel_initializer='he_normal')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def create_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.4)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(0.5)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    u2 = layers.UpSampling2D(size=(2, 2))(conv3)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.4)(conv4)

    u1 = layers.UpSampling2D(size=(2, 2))(conv4)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.3)(conv5)


    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model

def get_psf(ind, freq_array, psf_array):
    freq = freq_array[ind]
    psf = psf_array[ind]
    return psf*freq

def data_generator(X_train, index_train, y_train, freq_array, psf_array, batch_size=16):
    while True:
        for i in range(0,len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            index_batch = index_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_batch = y_batch[...,np.newaxis]

            psf_batch = [get_psf(ind, freq_array, psf_array) for ind in index_batch]
            combined_input_batch = np.stack((X_batch, np.array(psf_batch)), axis=-1)

        yield combined_input_batch, y_batch

def create_two_stage_model(input_shape_dirty, input_shape_psf, input_shape_freq):
    inputs_dirty = tf.keras.Input(shape=input_shape_dirty)
    inputs_psf = tf.keras.Input(shape=input_shape_psf)
    inputs_freq = tf.keras.Input(shape=input_shape_freq)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs_psf)
    encoded_psf = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Concatenate(axis=-1)([inputs_dirty, encoded_psf, input_freq])

    return models.Model(inputs=inputs, outputs=outputs)

def main(config_name):
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU instead.")
    config = configparser.ConfigParser()
    config.read(config_name)
    freq_start = float(config['INPUT']['freq_start'])
    freq_step = float(config['INPUT']['freq_step'])
    num_freq = int(config['INPUT']['num_freq'])
    freq_array = np.arange(freq_start, freq_start + freq_step * (num_freq-1) + 0.0001, freq_step)
    data_path = config['INPUT']['data_path']
    dirty_uniform = load_data(config['INPUT']['dirty_uniform'])
    dirty_natural = load_data(config['INPUT']['dirty_natural'])
    dirty_briggs = load_data(config['INPUT']['dirty_briggs'])
    psf_uniform = load_data(config['INPUT']['psf_uniform'])
    psf_natural = load_data(config['INPUT']['psf_natural'])
    psf_briggs = load_data(config['INPUT']['psf_briggs'])
    realdata = load_data(config['INPUT']['real_data'])
    FLAG_PSF = False
#    dirty_use = np.concatenate((dirty_uniform, dirty_natural, dirty_briggs), axis=0)
    dirty_use = dirty_briggs
    realdata_use = realdata
    psf_array_use = psf_briggs
    freq_array_use = freq_array
#    realdata_use = np.concatenate((realdata, realdata, realdata), axis=0)
#    freq_array_use = np.concatenate((freq_array, freq_array, freq_array), axis=0)
    Xtrain, Xval, Xtest, freq_train, freq_val, freq_test, ytrain, yval, ytest = preprocess_images(dirty_use, realdata_use, freq_array_use,128,100)
    print(Xtrain.shape, Xval.shape, Xtest.shape, freq_train.shape, freq_val.shape, freq_test.shape, ytrain.shape, yval.shape, ytest.shape)
    psf_use = psf_uniform
    model = cdae_model((128, 128, 1))
    huber_loss = tf.keras.losses.Huber(delta=1000)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('cdae.h5', monitor='val_loss', verbose=1, save_best_only=True)

    batch_size = 32
    if FLAG_PSF:
        steps_per_epoch = len(Xtrain)//batch_size
        val_steps = len(Xval)//batch_size
        train_gen = data_generator(Xtrain, freq_train, ytrain, freq_array, psf_use, batch_size=batch_size)
        val_gen = data_generator(Xval, freq_val, yval, freq_array, psf_use, batch_size=batch_size)
        test_gen = data_generator(Xtest, freq_test, ytest, freq_array, psf_use, batch_size=batch_size)

    history = model.fit(Xtrain, ytrain, batch_size=batch_size, epochs=100, validation_data=(Xval, yval), callbacks=[early_stopping, model_checkpoint])
#    history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=10 , validation_data=val_gen, validation_steps=val_steps, callbacks=[early_stopping, model_checkpoint])

    model.save('deconv_final.h5')
    test_loss, test_mse = model.evaluate(Xtest, ytest, verbose=-1)
#    test_loss, test_mse = model.evaluate_generator(test_gen, steps=len(Xtest)//batch_size, verbose = -1)
    print(test_loss, test_mse)

if __name__ == '__main__':
    config_name = sys.argv[1]
    main(config_name)
