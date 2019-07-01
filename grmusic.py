import keras
import sys
from midi_data_RNN import notetomidi
import os
import numpy as np
from random import choice



def generation(model,inp,inpspace):
    model = keras.models.load_model(model)
    print (model.summary())
    inputspace = np.load(inpspace)

    sample = choice(inputspace)
    output = sample
    output = list(output)
    sample = np.array(sample)
    window = output
    window = list(window)
    




    #return
    for i in range(256):
        new_input = sample.reshape(1,127)
        new_input = new_input
        next_note = model.predict(new_input)
        #print (len(next_note[0]))
        next_note = np.random.choice(38,1,p=next_note[0])

        print (next_note)
        #return
        output.append(int(next_note))
        
        window = output[i+1:]
        #print (len(window))
        sample = np.array(window)


        



    
    print (output)
    output = np.array(output)
    notetomidi(output,0)
    #print (model.predict_classes(asdf))
    #print ("hellow")

if __name__ == '__main__':
    generation(sys.argv[1],sys.argv[2],sys.argv[3]) #input model init_array


    '''    
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