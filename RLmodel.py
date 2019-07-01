#-*- coding: utf-8 -*
import sys, os
import numpy as np
import random
from random import choice
import random
import keras
from midi_data_RNN import notetomidi
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import Dense,Embedding, Dropout, LSTM , Bidirectional , Activation
from keras.layers.normalization import BatchNormalization
from keras.initializers import Orthogonal,RandomNormal
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from collections import deque




class music_env(object):
    
    
    def __init__(self,space_len,RNN_model,Style_model,initnpy):
        print ("hello")
        self.space_len = space_len
        self.RNN_model = RNN_model
        self.Style_model = Style_model
        self.initnpy_path = initnpy
        self.action_space = [i for i in range(0,38)]  # 0 for rest 1 for nothing 2 : 37 c3 to b5
#------------------------- hand input ----------------------------# 
        self.action_size = 38           #action number 0~37

        self.init_space = np.load(initnpy)
      
    
    def action_space_num(self):
        return 38
    def action_space_sample(self): #chose a action
        return choice(list(self.action_space))

    def reset(self):
        #init_s_l = len(self.init_space)

        return choice(self.init_space)
    

    def class_reward(self,cur_state,action):
        #print (len(cur_state))
        
        predict_input = cur_state[1:]#.append(action) #+action
        predict_input.append(action)
        #predict_input = list(predict_input)
        
        #print (len(predict_input))
        style_label = cur_state[0]

        predict_input = np.array(predict_input).reshape((1,128))
        result = self.Style_model.predict(predict_input)[0]
        reward = result[style_label]  #get point
        reward = reward * 200
        return reward  
    
    def RNN_note_reward(self,cur_state,action):
        input_s = cur_state[1:]

        input_s = np.array(input_s).reshape(1,len(input_s))
        result = self.RNN_model.predict(input_s)[0]
        reward = result[action]
        reward = reward * 100
        #print ("RNN_note_done")
        return reward 
        
    def state_shape(self):    #maybe large style note will be good result????
        return 129   #state[0] for class state[1:] for music note
    
    def action_num(self):
        return self.action_size

#-----------------------------
    def if_in_same_tone(self,melody): # delete 0 and 1 
        det_m = set(melody)
        
        return 1
        

    def if_cont_note(self,melody):    # only check last note?
        cont_count = 0     # check last not 1 note
        return 1

    def big_move(self,melody):       # if big move and same way 
        # find last not 1 note 
        f_last_note = 0
        s_last_note = 0
        for i in range(len(melody)-1,0,-1):
            if melody[i] > 1 and f_last_note == 0:
                f_last_note = melody[i]
            elif melody[i] > 1 and s_last_note == 0 and f_last_note > 1:
                s_last_note = melody[i]
            elif f_last_note > 1 and s_last_note > 1 :
                break

        hop_reward = abs(f_last_note - s_last_note)
        if hop_reward > 10 :
            hop_reward = hop_reward * hop_reward * (-1)
        else :
            hop_reward = 10

        return hop_reward
    
    def if_note_too_long(self,melody):
        #if cont 1 is too many
        one_count = 0
        for i in range(len(melody)-1,0,-1):
            if melody[i] == 1:
                one_count += 1
            else :
                break
        if one_count > 7:
            one_count = one_count * one_count * (-1)

        return one_count

    def melody_similarity(self,melody):
        m_l = len(melody)
        a = melody[(m_l/2):(m_l)]
        b = melody[(m_l/2-m_l/4):(m_l-m_l/4)]
        #----reward between 0~1 ----#
        similary_reward = stats.pearsonr(a,b)   #Correlation coefficient 
        similary_reward -= 0.5 #____ -0.5 ~ 0.5 _____
        similary_reward  = similary_reward * (-1) #if high get low reward
        return similary_reward * 10  # get -5 ~ 5 point
#--------------------
    def calculate(self,sp,cn,bm,tl,st,cr,nr):
        return sp+cn+bm+tl+st+cr+nr
    def step(self,cur_state,action):
    #--------------- Network reward -------------------#
        cr = self.class_reward(cur_state,action)
        nr = self.RNN_note_reward(cur_state,action)
        melody = cur_state.copy()#.append(action) #cur_state has been add action
        melody.append(action)
    #-------------- music theory point ----------------#
        sp = self.melody_similarity(melody)
        cn = self.if_cont_note(melody)
        bm = self.big_move(melody)
        tl = self.if_note_too_long(melody)
        st = self.if_in_same_tone(melody)
        
        new_state = cur_state[2:].copy()         #[style state] slide window
        new_state.insert(0,cur_state[0])  #add style_code in head
        #print (len(new_state))
        #sys.exit(1)
        
        reward = self.calculate(sp,cn,bm,tl,st,cr,nr)
        done = False
        return new_state,reward,done


    #def get_action(cur_state):
'''
class music_theory_reward(object):
    def __init__(self,cur_state,action): 
        self.state = cur_state
        self.action = action
'''





class DQN:
    def __init__(self, env):
        self.env     = env
#--------------------------------------------
        self.memory  = deque(maxlen=2000)    
        self.gamma = 0.85
        self.out_d = 38 * 7
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.input_state_len = 128
        self.seed = 9527
#--------------------------------------------
        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        #state_shape  = self.env.state_shape()
        model.add(Embedding(input_dim=self.env.action_num(),
               output_dim=self.out_d,        
               #input_shape = (38,255)
               input_length = self.input_state_len,
               ))
        model.add(Bidirectional(LSTM(512,
                      activation = "tanh",
                      #recurrent_activation = "hard_sigmoid",
                      use_bias = True,
                      #bias_initializer="ones",
                      recurrent_initializer = "orthogonal",
                      kernel_initializer = "glorot_uniform",
                      recurrent_dropout = 0.7
                      )))
        model.add(Dropout(0.7))
       #model.add(Dense(48, activation="relu"))
       #model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_num(),
                 #use_bias = True,                    
                 kernel_initializer = RandomNormal(mean=0.0,
                                           stddev=0.15,seed=self.seed),#Orthogonal(gain=1.0,seed=seed)
                 #kernel_regularizer = l2(l2_n)
                 #activation="softmax"
                ))  
        model.compile(loss="categorical_crossentropy",
            optimizer=Adam(lr=self.learning_rate)) 

        return model

    def act(self, state):

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        state = np.array(state).reshape(1,128)
        if np.random.random() < self.epsilon:
            return self.env.action_space_sample()

        return np.argmax(self.model.predict(state)[0]) # choice predict note

    def remember(self, state, action, reward, new_state, done):
        #print (len(state))
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):

        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            #print (len(state))
            state = np.array(state).reshape(1,128)
            new_state = new_state[0:128]
            new_state = np.array(new_state).reshape(1,128)
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward

            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=1)

    def target_train(self):


        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def save_model(self, fn):

        self.model.save(fn)

def make_init():
    a = [0 for i in range(64)]
    b = [7,1,1,1,8,1,1,1,9,1,1,1,14,1,1,15,1,1,17,18,19,21,19,1,1,1,14,1,1,13,1,1,14,1,1,15,1,1,17,22,1,1]
    #print (len(a))
    print (len(b))
    c = a+b+b
    c = c + [7,1,8]
    c = c[:127]
    return c

def main(RNN_model,Style_model,initnpy):

    
    env     = music_env(127,RNN_model,Style_model,initnpy)#gym.make("MountainCar-v0")
    gamma   = 0.9
    epsilon = .95
    trials  = 10
    trial_len = 512
    cur_state = []#make_init()  ## init state
    #init_style = random.randint(0,3)  # [classical , jazz , hymn , vgm]
    # updateTargetNetwork = 1000
    f_m_list = []
    final_music = []
    

    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        init_style = random.randint(0,3)
        cur_state = env.reset()  #.reshape(1,len(cur_state))
        cur_state = list(cur_state)
        final_music = cur_state.copy()
        cur_state.insert(0,init_style)
        #print (len(cur_state))
        for step in range(trial_len):
            print ("----- step %d -----"%(step))
            action = dqn_agent.act(cur_state)
            #print (cur_state)
            final_music.append(action)
            #print (cur_state)
            #print (cur_state)
            new_state, reward, done = env.step(cur_state,action)
            #print (cur_state)
            cur_state = cur_state[0:128] # why cur_state add action?


            if step == trial_len - 1 :
                done = True
            #print (new_state)
            dqn_agent.remember(cur_state, action, reward, new_state, done)    
            print ("_____replay_____")
            dqn_agent.replay()       # internally iterates default (prediction) model
            print ("__target_train__")
            dqn_agent.target_train() # iterates target model
            cur_state = new_state

        '''
       if step >= 199:
           print("Failed to complete in trial {}".format(trial))
           if step % 10 == 0:
               dqn_agent.save_model("trial-{}.model".format(trial))
        

       else:
           print("Completed in {} trials".format(trial))
           dqn_agent.save_model("success.model")
           break
        '''
        f_m_list.append(final_music)
        print (f_m_list)
        notetomidi(f_m_list,trial)
        dqn_agent.save_model("../LRcheckpoint/trial-{}.model".format(trial))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print ("enter python3 RLmodel.py \"rnnmodelpath\" \"stylemodel\" \"initnpy\" ")
        sys.exit(1)
        
    RNN_model = keras.models.load_model(sys.argv[1])  #argv[1] for RNN model
    Style_model = keras.models.load_model(sys.argv[2]) #argv[2] for Style Model

    main(RNN_model = RNN_model,Style_model = Style_model, initnpy = sys.argv[3])