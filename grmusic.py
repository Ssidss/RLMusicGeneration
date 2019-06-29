import keras
import sys
from midi_data_RNN import notetomidi
import os
import numpy as np



def generation(model,inp):
    model = keras.models.load_model(model)
    print (model.summary())
    asdf = []
    input_a = np.load(inp)
    input_a = input_a[:127]
    a = np.array(input_a) 
    b = a.tolist()
    a = [a,]
    asdf.append(a)
    #for i in len(asdf):
    #    asdf[i] = asdf[i]/37
    #b = a
    #a.reshape(255,1)
    
    #asdf.append(input_a)
    #print (model.predict_classes(asdf))
    #return
    #next_note = model.predict_classes(a)
    #print (next_note)
    #input_a = [input_a,]
    #asdf.append(input_a)
    #asdf.append(input_a) 
    #print(asdf[0])
    #print (asdf[0])
    #print(asdf[0:])
    
    #return
    print(asdf)
    aa = model.predict(asdf)

    #print (len(aa))
    aa = np.array(aa)
    '''
    print (str(aa))
    bb = str(aa).split(" ")
    print (len(bb))
    prob = []
    for ax in bb:
        if len(ax) > 1:
            print (ax)
            prob.append(ax)

    print (len(prob))
    '''
    #return
    for i in range(128):
        next_note = model.predict_classes(asdf)
        print (next_note)
        #return
        b.append(int(next_note))

        a = [b[i+1:],]
        asdf = []
        #a = a/37
        asdf.append(a)

        #print (asdf)
        #input_a[0] += next_note
        #asdf[0] = 
    
    print (b)
    b = np.array(b)
    notetomidi(b,0)
    #print (model.predict_classes(asdf))
    #print ("hellow")

if __name__ == '__main__':
    generation(sys.argv[1],sys.argv[2]) #input model init_array
