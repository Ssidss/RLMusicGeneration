#-*- coding: utf-8 -*
import numpy as np
from keras import backend as Ki
import random
import keras
import sys
import os
from keras.callbacks import ModelCheckpoint
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout,Bidirectional,Activation
from keras.optimizers import Adam,SGD
from keras.layers import Dense,Embedding, Dropout, LSTM, Input
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from midi_data import notetotrain,notetomidi
from sklearn.model_selection import train_test_split
from keras.regularizers import l2


def cmodel(path):
    
    l2_n = 0.25
    learning_rate = 0.001
    batch_s = 32 
    epoch = 40
    input_dime = 38
    #train_x += test_x
    #train_y += test_y
    x = []
    y = []
    pathdir = os.listdir(path)
    i = 0
    label = []
    
    for d in pathdir:
        print ("in %s / %s"%(path,d))
        xt = notetotrain(path + d)
        label.append(str(d))
        print (len(xt))
        x += xt
        y = y+[ i for j in range(len(xt))]
        #y += label
        i += 1 
    
    #print (len(x))
    #print (len(y))
    #print (y)
    print (label)
    tray = []
    train_x , test_x, train_y ,test_y = train_test_split(x,y, test_size = 0.1,random_state = 1)
    for ia in range (i):
        tray.append(test_y.count(ia))
    print (tray)
    train_x = np.array(train_x)/37
    train_y = np.array(train_y)
    test_x  = np.array(test_x)/37
    test_y  = np.array(test_y)
    

    '''
    datapath = "./edata/"
    train_x = np.load(datapath+"trainx.npy")#,train_x)
    train_y = np.load(datapath+"trainy.npy")#,train_y)
    test_x =  np.load(datapath+"testx.npy")#,test_x)
    test_y =  np.load(datapath+"testy.npy")#),test_y)
    #np.load(
    '''
    train_y  = to_categorical(train_y,i)
    test_y  = to_categorical(test_y,i)
    #print (train_y[0])
    #print (train_x[0])
    num_class = i
    #print (i)
    #print (num_class)
    #print (len(train_y))
    #print (len(train_y[0]))
    #print (train_y[0])
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    o_d = 38 * 100
    model.add(Embedding(input_dim=38,output_dim=o_d,input_length=128))
    #model.add(Bidirectional(LSTM(1024,return_sequences=True),input_shape=(52,1)))
    
    model.add(Bidirectional
            (LSTM(2048,
                            activation = "tanh",
                            use_bias = True,
                            #recurrent_initializer = "orthogonal",
                            kernel_initializer = "glorot_uniform",
                            recurrent_dropout = 0.9)))
    model.add(Dropout(0.9))

    '''
    model.add(Dense(512,
                    kernel_initializer = keras.initializers.random_normal(stddev=1,seed=99),
                    kernel_regularizer = l2(l2_n),
                    activation="relu")
                    )
    model.add(Dense(512,
                    kernel_initializer = keras.initializers.random_normal(stddev=1,seed=1),
                    kernel_regularizer = l2(l2_n),
                    activation="relu")
                    )
    model.add(Dense(512,
                    kernel_initializer = keras.initializers.random_normal(stddev=1,seed=2),
                    kernel_regularizer = l2(l2_n),
                    activation="relu")
                    )
    '''
    model.add(Dense(i,
                    kernel_initializer = keras.initializers.random_normal(stddev=1,seed=3),
                    kernel_regularizer = l2(l2_n)
                    #activation="softmax")
                    ))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))
    print(model.summary())
    #adam = Adam(learning_rate)
    checkpath = "../RNNcheckpoint/esaved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    model.compile(loss='categorical_crossentropy',
            optimizer= Adam(lr=learning_rate,decay = 0.01),#SGD(lr=learning_rate,decay = 1e-5,
                #momentum=0.9,nesterov=True),#'adam',
            metrics=['acc'])
    #model = keras.models.load_model("./emodel.h5") 
    checkpoint = ModelCheckpoint(checkpath,
            monitor='val_acc',
            verbose=1,
            save_best_only = True, 
            mode = 'max')
    callbacks_list = [checkpoint]
    model.fit(train_x,train_y,
            batch_size=batch_s,
            callbacks=callbacks_list,
            epochs=epoch,
            verbose=1,
            validation_data=(test_x,test_y),
            shuffle = True)
    #Ki.clear_session()
    mp = "./emodel.h5"
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








    


