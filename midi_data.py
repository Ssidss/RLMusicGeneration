import midi
import numpy as np
import sys
import os


lowerBound = 36   #C3 36
upperBound = 71  #B7 71
tones = {'C3' :  2,
         'C#3':  3, 
         'D3' :  4,
         'D#3':  5,
         'E3' :  6,
         'F3' :  7,
         'F#3':  8,
         'G3' :  9,
         'G#3':  10,
         'A3' :  11,
         'A#3':  12,
         'B3' :  13}
    
# %  => pitch 
# // => Range
base_tones = {'0':'C',
              '1':'C#', 
              '2':'D',
              '3':'D#',
              '4':'E',
              '5':'F',
              '6':'F#',
              '7':'G',
              '8':'G#',
              '9':'A',
              '10':'A#',
              '11':'B'}

Melody_name = [ "main","vocal","guit","lead","guitar","vocal",
                "piano","sax","solo","melody","rhythm","think","string",
                "harmonica","song","gtr","gt","music"]                
                       


#len(note)
# 32:16 、64:8、128:4、256:2、512:1、192、216、240、360
# n/32 
#traininput dim = 128 


def notelen(note_tick,A):
    for i in range(min(note_tick//32,3)) :
        A.append(1)
        #A+=int(1)
    return A

def notetotrain():
    lf = open("newcluster.txt")
    p = 127
    labelname = lf.read()
    labels = []
    labels[:] = labelname.split("\n")
    labels.pop()
    data  = []
    label = []
    maxlen = 511
    minlen = maxlen / 2
    midilist = os.listdir("./MIDIs")
    midilist.sort()
    datapath = "./MIDIs"
    labelp = 0
    #for lab in labels:
    for mid in midilist:
        
        midata = os.path.join(datapath,mid)
        midinote = miditonote(midata)
        lname = labels[labelp]
        #print ("song: %s have %d data its label is: %s"%(str(mid),len(midinote),lname))
        for mm in midinote:
            md = mm
            while (len(md)>0): 
                if md[0] == 1 or md[0] == 0:
                    md = md[1:]
                    continue
                if len(md) < minlen:
                    break
                    #md = []
                elif len(md) >= maxlen :
                    mdp = md[0:maxlen]
                    mdp.append(int(md[0]))
                    #mdp += md[0]
                    data.append(mdp)
                    label.append(lname)
                    md = md[maxlen:]
                elif len(md)>minlen and len(md)<maxlen:
                    mdp = md
                    mdp += md[0:(maxlen-len(md))]
                    mdp.append(int(md[0]))
                    #mdp += md[0]
                    data.append(mdp)
                    label.append(lname)
                    break
                    #md = []
        labelp += 1
    print ("training data done")

    labelss = np.zeros([len(label),5])
    for i in range(len(label)):
        
        j = int(label[i][-1])-1
        labelss[i][j] = 1
    
    
    return data,labelss
    #lset = set(label)
    #c1 = label.count(lset[0])
    #c2 = label.count(lset[1])
    #c3 = label.count(lset[2])
    #c4 = label.count(lset[3])
    #c5 = label.count(lset[4])

    #print ("c1 %d ,c2 %d ,c3 %d ,c4 %d ,c5 %d"%(c1,c2,c3,c4,c5))
    #for ss in lset:
        #print("%s have %d data"%(str(ss),label.count(ss)))
    #print (len(label),len(data))
    
    #datalen = 0
    #for dd in data:
    #    datalen += len(dd)

    #print (datalen/len(data))
        

def miditypedetech(midipath):    
    path = os.path.join("./",midipath)
    f = open("./mididatatitle",'w')
    K = os.listdir(path)
    K.sort()
    print (len(K))
    for data in K:
        datapath = os.path.join(path,data)
        #print (data,end=':')
        f.write(data + " : ")
        pattern = midi.read_midifile(datapath)
        
        for i in pattern:
            #print (i)
            for j in i:
                if isinstance(j, midi.TrackNameEvent):                    
                    #print (j.text,end=' , ')
                    f.write(j.text + " , ")
        f.write("\r\n")
        #print ("")
    
def find_melody(pattern):
    melodymidi = []
    if len(pattern) < 3:
        return pattern
    for j in pattern:
        for tra in j:
            if isinstance(tra, midi.TrackNameEvent):
                for mstr in Melody_name:
                    if mstr in str(tra.text).lower():
                        melodymidi.append(j)
                        break
    return melodymidi                        

def miditonote(midifile):    
    first_note = True
    pattern = midi.read_midifile(midifile)
    
    res = []
    last_note = -1
    melodymidi = find_melody(pattern)    
    for j in melodymidi: 
        #print (i)
        output = []   
        for i in j :    
            #if isinstance(i, midi.TrackNameEvent):
                #print (i.text)     
            if isinstance(i, midi.NoteEvent):
                if i.pitch == last_note:
                    output = notelen(i.tick,output)
                    last_note = -1
                    continue
                elif i.velocity == 0 or i.tick == 0:
                    if first_note :
                        last_note = i.pitch
                        output = notelen(i.tick,output)
                        output.append(int(i.pitch))
                        #output += int(i.pitch)
                        #t_note  =  base_tones [str(i.pitch % 12)]
                        #t_range =  i.pitch // 12
                        #output.append(str(t_note)+str(t_range))
                        first_note = False
                    continue
                if (i.pitch < lowerBound) or (i.pitch >= upperBound) :
                    output.append(int(i.pitch%12 + 48)  )  
                    #output += int(i.pitch%12+24)
                    # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                else:
                    if isinstance(i, midi.NoteOffEvent):
                        output.append(int(0))
                        #output+= int(0)
                    else:
                        last_note = i.pitch
                        output = notelen(i.tick,output)
                        output.append(int(i.pitch))
                        #output += int(i.pitch)
                        #t_note  =  base_tones [str(i.pitch % 12)]
                        #t_range =  i.pitch // 12
                        #output.append(str(t_note)+str(t_range))
        #print (len(output))
        res.append(output)
        #print (output)
    return res

#f = open("./miditest.txt",'w')
#f.write(str(A))  

def notetomidi(notearray,i):
    note_len = 54

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    
    last_note = 0
    for n in notearray:
        if n == 1 :
            note_len = note_len + 54
        elif int(n) > 1 :   
            track.append(midi.NoteOnEvent(tick=note_len, channel=0, data=[last_note, 0]))         
            track.append(midi.NoteOnEvent(tick=0, channel=0, data=[n, 110]))            
            last_note = n
            note_len = 54
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("example"+str(i)+".mid",pattern)

            

'''
    pattern1 = midi.Pattern()
    track1 = midi.Track()
    pattern1.append(track1)
    for i in pattern[0]:
        track1.append(i)
    midi.write_midifile("test0.mid", pattern1)
'''
    
	
    #print(midi.NoteOnEvent(tick=0,velocity=50,pitch=midi.G_3))

if __name__ == '__main__':
    
    #if len(sys.argv) != 2:
    #    sys.stderr.write("usage: midi_data.py filename \n")
    #    sys.exit(-1)
    
    notetotrain()
    #A = miditonote(sys.argv[1])
    
    #for i in range(len(A)):
    #    notetomidi(A[i],i)

    #miditypedetech(sys.argv[1])



    '''
    print (A)
    f = open("./miditest.txt",'w')
    f.write(str(A))
    '''
