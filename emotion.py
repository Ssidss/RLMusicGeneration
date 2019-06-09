#-*- coding: utf-8 -*
import numpy as np
import tensorflow as tf
import sys
import os
from midi_data import notetotrain 
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.utils.np_utils import to_categorical


def cmodel(train_x,train_y,test_x,test_y):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x  = np.array(test_x)
    text_y  = np.array(test_y)
    num_class = 5
    #train_y = to_categorical(train_y,num_class)
    #test_y  = to_categorical(test_y,num_class)
    model = Sequential()
    model.add(Embedding(input_dim=128,output_dim=1,input_length=128))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256,return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_class,activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.fit(train_x,train_y,batch_size=512,epochs=20,verbose=1,validation_data=(test_x,test_y))
     

def train_test():
    data = []
    label = []
    data ,label = notetotrain()
    td = int(len(data)*0.9)
    train_x ,train_y = data[0:td],label[0:td]
    test_x  ,test_y  = data[td+1:len(data)-1] , label[td+1: len(label)-1]
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Enter python3 emotion.py \"mode\"")
        sys.exit(1)
    if sys.argv[1]=="test":
        print ("hello")
    elif sys.argv[1]=="train":
        tr_x,tr_y,te_x,te_y = train_test()
        cmodel(tr_x,tr_y,te_x,te_y)








    


