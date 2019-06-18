#-*- coding: utf-8 -*
import numpy as np
import tensorflow as tf
import sys
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.optimizers import rmsprop
from midi_data import notetotrain 
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional,TimeDistributed
import keras

def cmodel(path):
    
    learning_rate = 0.000025
    input_dime = 38
    #train_x += test_x
    #train_y += test_y
    x = []
    y = []
    pathdir = os.listdir(path)
    i = 0
    for d in pathdir:
        print ("in %s / %s"%(path,d))
        xt = notetotrain(path + d)
        print (len(xt))
        x += xt
        y = y+[ i for j in range(len(xt))]
        #y += label
        i += 1 
    
    #print (len(x))
    #print (len(y))
    #print (y)
    
    train_x , test_x, train_y ,test_y = train_test_split(x,y, test_size = 0.1,random_state = 42)
    train_x = np.array(train_x)/37
    train_y = np.array(train_y)
    test_x  = np.array(test_x)/37
    test_y  = np.array(test_y)
    train_y  = to_categorical(train_y,i)
    test_y  = to_categorical(test_y,i)
    print (train_y[0])
    print (train_x[0])
    num_class = i
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Embedding(input_dim=38,output_dim=512,mask_zero=True))
    #model.add(Bidirectional(LSTM(1024,return_sequences=True),input_shape=(52,1)))
    
    model.add(LSTM(1024,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(1024,return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(num_class,
          kernel_initializer=keras.initializers.random_normal(stddev=0.05),
          activation='softmax'))
    print(model.summary())
    adam = Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.fit(train_x,train_y,batch_size=512,epochs=4000,verbose=1,validation_data=(test_x,test_y))
    mp = "./model.h5"
    model.save(mp)
     

def train_test():
    data = []
    label = []
    data = notetotrain()
    td = int(len(data)*0.9)
    train_x ,train_y = data[0:td],label[0:td]
    test_x  ,test_y  = data[td+1:len(data)-1] , label[td+1: len(label)-1]
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ("Enter python3 emotion.py \"mode\" \"datapath\"  ")
        sys.exit(1)
    if sys.argv[1]=="test":
        print ("hello")
    elif sys.argv[1]=="train":
        #tr_x,tr_y,te_x,te_y = train_test()
        cmodel(sys.argv[2])








    


