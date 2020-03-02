from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM, GRU, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def get_base_model(n_classes):
    model = Sequential()
    model.add(Conv1D(input_shape=(187,1),filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(LSTM(64,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(n_classes,activation='softmax'))
    model.summary()
    return model, 'base_model'

def get_pure_LSTM(n_classes):
    model.add(LSTM(256,return_sequences=True,input_shape=(187,1)))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(64,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(n_classes,activation='softmax'))
    model.summary()
    return model, 'lstm_model'

def get_debug_model(n_classes):
    model = Sequential()
    model.add(Conv1D(input_shape=(187,1),filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(n_classes,activation='softmax'))
    model.summary()
    return model, 'debug_model'
