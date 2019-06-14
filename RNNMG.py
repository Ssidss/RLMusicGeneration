import gym
import numpy as np
import random
import keras
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import Dense,Embedding, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from collections import deque
from keras.utils.np_utils import to_categorical
from midi_data_RNN import noteto_RNNtrain,notetomidi
from sklearn.model_selection import train_test_split


def train(x,y):  
    # input padding  0000
    # output one hot ?  max len = 8 ==> [4,1,1,1,0,0,0,0]
    train_x , test_x, train_y ,test_y = train_test_split(x,y, test_size = 0.1,random_state = 42)
    learning_rate = 0.01
    
    train_x = (np.array(train_x))/37
    train_y = (np.array(train_y))
    test_x  = (np.array(test_x))/37
    test_y  = (np.array(test_y))
    '''
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x  = np.array(test_x)
    test_y  = np.array(test_y)
    '''
    train_y  = to_categorical(train_y,38)
    test_y  = to_categorical(test_y,38)
    print (len(train_x))
    print (len(train_x[0]))
    print (len(train_y))
    print ((train_y))
    
    #print (len(train_y[0]))
    #print (train_y)
    #return

    seed = 7
    np.random.seed(seed)

    model  = Sequential()
    model.add(Embedding(input_dim=38,output_dim=1024,mask_zero=True))
    model.add(Dropout(0.2))
    model.add(LSTM(1024,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(1024,return_sequences=False))
    model.add(Dense(38,
                    kernel_initializer = keras.initializers.random_normal(stddev=0.05),
                    activation="softmax"))  
    print(model.summary())                          
    adam = Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.fit(train_x,train_y,batch_size=32,epochs=1,verbose=1,validation_data=(test_x,test_y))
    mp = "./model.h5"
    model.save(mp)

    '''
    gx = [int (i) for i in train_x[0]]
    ty = [int (i) for i in train_y[0]]

    gy = model.predict(train_x,batch_size=None,verbose=0,steps=None)
    gy = gy[0]
    gy = [int (i) for i in gy[0]]
    print (gy)

    output1 = gx.append(ty)
    output2 = gx.append(gy)
    output1 = output1*37 + 35
    output2 = output2*37 + 35

    notetomidi(output1,1)
    notetomidi(output2,2)
    '''




    
def to_s2s(data):
    x = []
    y = []
    for d in data:
        x.append(d[:-1])
        y.append(d[-1])
        '''
        for i in temp_y:
            if i == 1 or i == 0:
                temp_y = temp_y[1:]
            else:
                break
        while len(temp_y) < 8:
            #temp_y += int (0)
            temp_y.append(0)

        y.append(temp_y)
        '''
        #y+= temp_y
    #print (len(x[0]))
    #print (len(y[0]))
    return x,y



def main():
    data = noteto_RNNtrain()
    x,y = to_s2s(data)
    train(x,y)
    #print ("Hello")

if __name__ == '__main__':
    main()
