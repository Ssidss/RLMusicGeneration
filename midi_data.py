import midi, numpy
import sys


lowerBound = 12  #C3
upperBound = 127  #B7
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

#len(note)
# 32:16 、64:8、128:4、256:2、512:1、192、216、240、360
# n/32 
def notelen(note_tick,A):
    for i in range(note_tick//32) :
        A.append(1)
    return A


def miditonote(midifile):
    first_note = True
    pattern = midi.read_midifile(midifile)
    output = []
    last_note = -1
    for i in pattern[1]:        
        #print (i)
        if isinstance(i, midi.NoteEvent):
            if i.pitch == last_note:
                output = notelen(i.tick,output)
                last_note = -1
                continue
            elif i.velocity == 0 or i.tick == 0:
                if first_note :
                    last_note = i.pitch
                    output = notelen(i.tick,output)
                    output.append(i.pitch)
                    #t_note  =  base_tones [str(i.pitch % 12)]
                    #t_range =  i.pitch // 12
                    #output.append(str(t_note)+str(t_range))
                    first_note = False
                continue
            if (i.pitch < lowerBound) or (i.pitch >= upperBound) :
                output.append(str(1))    
                # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
            else:
                if isinstance(i, midi.NoteOffEvent):
                    output.append(str(0))
                else:
                    last_note = i.pitch
                    output = notelen(i.tick,output)
                    output.append(i.pitch)
                    #t_note  =  base_tones [str(i.pitch % 12)]
                    #t_range =  i.pitch // 12
                    #output.append(str(t_note)+str(t_range))
    #print (len(output))
    return output
   

def notetomidi(notearray):
    note_len = 32

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    
    last_note = 0
    for n in notearray:
        if n == 1 :
            note_len = note_len + 32
        elif n > 1 :   
            track.append(midi.NoteOnEvent(tick=note_len, channel=0, data=[last_note, 0]))         
            track.append(midi.NoteOnEvent(tick=0, channel=0, data=[n, 110]))            
            last_note = n
            note_len = 32
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("example.mid",pattern)

            

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
    
    if len(sys.argv) != 2:
        sys.stderr.write("usage: main.py size_ \n")
        sys.exit(-1)
    
    A = miditonote(sys.argv[1])
    notetomidi(A)
    '''
    print (A)
    f = open("./miditest.txt",'w')
    f.write(str(A))
    '''
