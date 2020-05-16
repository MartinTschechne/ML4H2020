from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def get_CNN_model(seq_length):
    '''model_id: cnn-model'''
    model = Sequential()
    model.add(Conv1D(input_shape=(seq_length,1),filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    return model

def get_debug_model(seq_length):
    '''model_id: debug'''
    model = Sequential()
    model.add(Conv1D(input_shape=(seq_length,1),filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    return model

def get_model(model_id, seq_length):
    if model_id == 'cnn-model':
        return get_CNN_model(seq_length)
    elif model_id == 'debug':
        return get_debug_model(seq_length)
    else:
        print("Please specify a model.")
        exit(-1)
