import numpy as np
import json
import sys
import os
import tensorflow as tf
sys.path.append('/home/cedra/aseman rafat/paper code/time_series_augmentation')
sys.path.append('time_series_augmentation')
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.callbacks import EarlyStopping,History

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, GRU, Conv1D, Softmax, Embedding, Conv2D, MaxPooling2D, Flatten, Dropout



def LSTM_model(x_test, y, x_train, y_train):
    x_val, x_test, y_val, y_test = train_test_split(x_test, y, test_size=0.5, train_size=0.5)
    x_train = x_train.reshape(1600, 50, 24)
    y_train = y_train.reshape(1600, 16)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Create a model combining CNN and RNN operations
    lr = 0.0005
    batch_size = 64
    lstm1 = 512
    lstm2 = 256
    dropout = 0.6

    conv1d_gru_model = Sequential()
    conv1d_gru_model.add(LSTM(lstm1, dropout=dropout, return_sequences=True, input_shape=(None, 24)))
    # conv1d_gru_model.add(LSTM(256,  dropout=0.2,return_sequences=True))
    conv1d_gru_model.add(LSTM(lstm2, dropout=dropout))
    # Predict 10 steps ahead
    conv1d_gru_model.add(Dense(16, activation='softmax'))

    # Compile and train the model
    conv1d_gru_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    cback = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=15,
                          verbose=0, mode='auto', restore_best_weights=True)

    history = conv1d_gru_model.fit(x_train, y_train, batch_size=batch_size, epochs=50, verbose=0,
                                   validation_data=(x_val, y_val), callbacks=[cback])

    predictions = conv1d_gru_model.predict(x_test)
    full = 0
    right = 0
    is_equal = tf.equal(tf.argmax(y_test, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
    print(accuracy)


def CNN_model(x_test, y, x_train, y_train):
    x_val, x_test, y_val, y_test = train_test_split(x_test, y, test_size=0.5, train_size=0.5)
    x_train = x_train.reshape(1600, 50, 24)
    y_train = y_train.reshape(1600, 16)
    # CNN
    lr  =  0.0005
    batch_size = 64
    conv1 = 512
    conv2 = 512
    conv3 = 512
    dense = 256

    model = Sequential()
    model.add(Conv2D(conv1, (3, 2), activation='relu', input_shape=(50, 24, 1)))
    model.add(MaxPooling2D((3,2)))
    model.add(Conv2D(conv2, (3, 2), activation='relu'))
    model.add(MaxPooling2D((3,2)))
    model.add(Conv2D(conv3, (3, 2), activation='relu'))
    model.add(Flatten())

    model.add(Dense(dense, activation='relu'))
    model.add(Dense(16))
    cback=EarlyStopping(monitor='val_loss',min_delta=0,patience=15,verbose=0, mode='auto',restore_best_weights=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='mse',metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=80, validation_data=(x_val, y_val), verbose=0, callbacks = [cback], batch_size = batch_size)
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=0)
    print(test_acc)
