#Step 2 Build Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization
from keras.models import Sequential
import time
def create_model(batch_size,seq_len,input_dim,output_dim):
    model = Sequential()

    #model.add(LSTM(
    #    batch_input_shape=(batch_size,seq_len,input_dim),
    #    units=64,activation='relu',
    #    stateful=True,
    #    return_sequences=True))
    model.add(LSTM(
        input_shape=(seq_len,input_dim),
        units=64,activation='relu',
        return_sequences=True))
    
    model.add(LSTM(
        units=64,activation='relu',
        return_sequences=False))
    model.add(Dense(
        units=32))
    model.add(Activation('relu'))
    model.add(Dense(
        units=16))
    model.add(Activation('relu'))
    #model.add(Activation('relu'))
    #model.add(Dense(
    #    units=10))
    #model.add(Activation('relu'))
    #model.add(Dense(
    #    units=50))
    #model.add(Activation('tanh'))
    model.add(Dense(
        units=output_dim))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print ('compilation time : ', time.time() - start)
    return model