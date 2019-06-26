import numpy as np
import random
import keras
import sys
import os
from keras.callbacks import ModelCheckpoint
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout,Bidirectional,Activation
from keras.optimizers import Adam,SGD
from keras.layers import Dense,Embedding, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from midi_data_RNN import noteto_RNNtrain,notetomidi
from sklearn.model_selection import train_test_split
from keras.initializers import Orthogonal,RandomNormal
import matplotlib.pyplot as plt

def train(x,y):  
    # input padding  0000
    # output one hot ?  max len = 8 ==> [4,1,1,1,0,0,0,0]
    train_x , test_x, train_y ,test_y = train_test_split(x,y, test_size = 0.1)
    learning_rate = 0.1
    batch_s = 64
    epoch = 10
    
    tary = []
    for i in range (38):
        tary.append(train_y.count(i))
    print (tary)
    return

    train_x = (np.array(train_x))/37
    train_y = (np.array(train_y))
    test_x  = (np.array(test_x))/37
    test_y  = (np.array(test_y))
    '''
    tary = []
    for i in range (38):
        tary.append(train_y.count(i))
    print (tary)
    return
    '''
    #np.savetxt("./rnntrx.txt",train_x)
    #np.savetxt("./rnntex.txt",test_x)
    #np.savetxt("./rnntry.txt",train_y)
    #np.savetxt("./rnntey.txt",test_y)

    '''
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x  = np.array(test_x)
    test_y  = np.array(test_y)
    '''
    #train_x  = to_categorical(train_x,38)
    #test_x  = to_categorical(test_x,38)
    train_y  = to_categorical(train_y,38)
    test_y  = to_categorical(test_y,38)
    print (train_x)
    print (train_y)
    '''
    f = open("./y.txt",'w')
    for inn in train_y:
        f.write(str(inn))
    '''
    #return 
    #print (len(train_x))
    #print (len(train_x[0]))
    #print (len(train_y))
    #print ((train_y))
    
    #print (len(train_y[0]))
    #print (train_y)
    #return

    seed = 777
    np.random.seed(seed)
    out_d = 38 * 16
    model  = Sequential()
    model.add(Embedding(input_dim=38,
        output_dim=out_d,        
        input_length = 255,
        mask_zero=True))
    #model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(1024,
                      activation = "tanh",
                      #recurrent_activation = "hard_sigmoid",
                      #use_bias = True,
                      #bias_initializer="ones",
                      recurrent_initializer = "orthogonal",
                      kernel_initializer = "glorot_uniform",
                      recurrent_dropout = 0.3)))
    #model.add(Dense(512))
    model.add(Dropout(0.7))
    #model.add(LSTM(1024,return_sequences=False))
    model.add(Dense(38,
                    #use_bias = True,                    
                    kernel_initializer = RandomNormal(mean=0.0,
                        stddev=0.1,seed=seed),#Orthogonal(gain=1.0,seed=seed)
                    #activation="softmax"
                    ))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))
    print(model.summary())                          
    #adam = Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',
            optimizer= Adam(lr = learning_rate,decay = 0.1),#SGD(lr=learning_rate,decay = 1e-5,
                #momentum=0.9,nesterov=True),#'adam',
            metrics=['acc'])
    checkfile = "../RNNcheckpoint/Rweights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkfile, monitor = 'val_acc', verbose = 1,save_best_only = True,mode='max')
    #model = keras.models.load_model("./rnnmodel.h5")
    call_back = [checkpoint]
    model.fit(train_x,train_y,
            batch_size=batch_s,
            callbacks = call_back,
            epochs=epoch,
            verbose=1,
            validation_data=(test_x,test_y),
            shuffle = True)
    #print(history.history.key())
    # summarize history for accuracy
    '''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    '''
    mp = "./rnnmodel.h5"
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




