import numpy as np
import random
import keras
import sys
import os
from keras.callbacks import ModelCheckpoint
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout,Bidirectional
from keras.optimizers import Adam,SGD
from keras.layers import Dense,Embedding, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from midi_data_RNN import noteto_RNNtrain,notetomidi
from sklearn.model_selection import train_test_split

def train(x,y):  
    # input padding  0000
    # output one hot ?  max len = 8 ==> [4,1,1,1,0,0,0,0]
    train_x , test_x, train_y ,test_y = train_test_split(x,y, test_size = 0.1)
    learning_rate = 0.000025
    batch_s = 512 
    epoch = 40
    
    train_x = (np.array(train_x))
    train_y = (np.array(train_y))
    test_x  = (np.array(test_x))
    test_y  = (np.array(test_y))

    np.savetxt("./rnntrx.txt",train_x)
    np.savetxt("./rnntex.txt",test_x)
    np.savetxt("./rnntry.txt",train_y)
    np.savetxt("./rnntey.txt",test_y)

    '''
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x  = np.array(test_x)
    test_y  = np.array(test_y)
    '''
    train_y  = to_categorical(train_y,38)
    test_y  = to_categorical(test_y,38)
    #print (len(train_x))
    #print (len(train_x[0]))
    #print (len(train_y))
    #print ((train_y))
    
    #print (len(train_y[0]))
    #print (train_y)
    #return

    seed = 7
    np.random.seed(seed)

    model  = Sequential()
    model.add(Embedding(input_dim=38,
        output_dim=512,        
        mask_zero=True))
    #model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    #model.add(Dense(512))
    model.add(Dropout(0.7))
    #model.add(LSTM(1024,return_sequences=False))
    model.add(Dense(38,
                    kernel_initializer = keras.initializers.random_normal(stddev=0.01,seed=seed),
                    activation="softmax"))  
    print(model.summary())                          
    #adam = Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',
            optimizer= Adam(learning_rate),#SGD(lr=learning_rate,decay = 1e-5,
                #momentum=0.9,nesterov=True),#'adam',
            metrics=['acc'])
    checkfile = "./RNNcheckpoint/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkfile, monitor = 'val_acc', verbose = 1,save_best_only = True,mode='max')
    model = keras.models.load_model("./model.h5")
    model.fit(train_x,train_y,
            batch_size=batch_s,
            epochs=epoch,
            verbose=1,
            validation_data=(test_x,test_y),
            shuffle = True)
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
    print ("tos2s")
    for d in data:
        x.append(d[:-1])
        y.append(d[-1])
        #y+= temp_y
    #print (len(x[0]))
    #print (len(y[0]))
    return x,y



def main(midipath):
    data = []
    midilist = os.listdir(midipath)
    for mdir in midilist:
        mpath = os.path.join(midipath,mdir)
        data += noteto_RNNtrain(mpath)
    x,y = to_s2s(data)
    train(x,y)
    #print ("Hello")

if __name__ == '__main__':
    print ("Hello")
    if len(sys.argv) != 3:
        print ("enter python3 RNNMG.py  \"mode\"  \"midifiledir\" ")
        sys.exit(1)

 
    main(sys.argv[2])




