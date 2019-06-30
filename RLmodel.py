import gym
import sys, os
import numpy as np
import random
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import Dense,Embedding, Dropout, LSTM , Bidirectional , Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from collections import deque




class music_env(object):
    
    
    def __init__(self,space_len,RNN_model,Style_model):
        print ("hello")
        self.space_len = space_len
        self.RNN_model = rnnmodel
        self.Style_model = Style_model
        self.action_space = [i for i in range(0,38)]  # 0 for rest 1 for nothing 2 : 37 c3 to b5
#------------------------- hand input ----------------------------# 
        self action_size = 38           #action number 0~37
      
    def action_space():
        def num():
            return 38
        def sample(): #chose a action
            return choice(self.action_space)
    

    def class_reward(cur_state,action):
        predict_input = cur_state[1:]+action
        result = self.Style_model.predict(predict_input)
        reward = result[action]  #get point
        return reward  
    
    def RNN_note_reward(cur_state,action):
        input_s = cur_state[1:]
        result = self.RNN_model.predict(input_s)
        reward = result[action]
        return reward 
        
    def state_shape():    #maybe large style note will be good result????
        return 129   #state[0] for class state[1:] for music note
    
    def action_num()
        return action_size

#-----------------------------
    def if_in_same_tone(): # delete 0 and 1 
        

    def if_cont_note():    # only check last note?
        cont_count = 0     # check last not 1 note
        

    def big_move():       # if big move and same way 
        move_reward = 0   # find last not 1 note 
    
    def if_note_too_long():
        long_note_reward = 0  #if cont 1 is too many

    def melody_similarity():
        stats.pearsonr(a,b)   #Correlation coefficient
#--------------------
    def calculate(sp,cn,bm,tl,st,cr,nr):
        return sp+cn+bm+tl+st+cr+nr
    def step(cur_state,action):
        melody = cur_state + action
    #-------------- music theory point ----------------#
        sp = self.melody_similarity(melody)
        cn = self.if_cont_note(melody)
        bm = self.big_move(melody)
        tl = self.if_note_too_long(melody)
        st = self.if_in_same_tone(melody)
        cr = self.class_reward(cur_state,action)
        nr = self.RNN_note_reward(cur_state,action)
        new_state = cur_state[0]+cur_state[2:]+action
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
       self.input_state_len = 64
#--------------------------------------------
       self.model        = self.create_model()
       self.target_model = self.create_model()

   def create_model(self):
       model   = Sequential()
       state_shape  = self.env.state_shape()
       model.add(Embedding(input_dim=self.env.action_num(),
              output_dim=self.out_d,        
              #input_shape = (38,255)
              input_length = self.input_state_len,
              ))
       model.add(Bidirectional(LSTM(64,
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
                                           stddev=0.15,seed=seed),#Orthogonal(gain=1.0,seed=seed)
                 #kernel_regularizer = l2(l2_n)
                 #activation="softmax"
                ))  
       model.compile(loss="categorical_crossentropy",
           optimizer=Adam(lr=self.learning_rate)) 

       return model

   def act(self, state):

       self.epsilon *= self.epsilon_decay
       self.epsilon = max(self.epsilon_min, self.epsilon)
       if np.random.random() < self.epsilon:
           return self.env.action_space.sample()

       return np.argmax(self.model.predict(state)[0]) # choise predict note

   def remember(self, state, action, reward, new_state, done):

       self.memory.append([state, action, reward, new_state, done])

   def replay(self):

       batch_size = 32
       if len(self.memory) < batch_size: 
           return

       samples = random.sample(self.memory, batch_size)
       for sample in samples:
           state, action, reward, new_state, done = sample
           target = self.target_model.predict(state)
           if done:
               target[0][action] = reward

           else:
               Q_future = max(self.target_model.predict(new_state)[0])
               target[0][action] = reward + Q_future * self.gamma

           self.model.fit(state, target, epochs=1, verbose=0)

   def target_train(self):

       weights = self.model.get_weights()
       target_weights = self.target_model.get_weights()
       for i in range(len(target_weights)):
           target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

       self.target_model.set_weights(target_weights)

   def save_model(self, fn):

       self.model.save(fn)

def main(RNN_model,Style_model):

    
    env     = music_env(RNN_model,Style_model)#gym.make("MountainCar-v0")
    gamma   = 0.9
    epsilon = .95
    trials  = 1000
    trial_len = 500
    cur_state = []  ## init state
   # updateTargetNetwork = 1000

    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1,2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(cur_state,action)
            # reward = reward if not done else -20
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)        
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model
            cur_state = new_state
            if done:
                break
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("enter python3 RLmodel.py \"rnnmodelpath\" \"stylemodel\"   ")
        sys.exit(1)
    RNN_model = keras.models.load_model(argv[1])  #argv[1] for RNN model
    Style_model = keras.models.load_model(argv[2]) #argv[2] for Style Model

    main(RNN_model = RNN_model,Style_model = Style_model)