#!/usr/bin/env python2
# -*- coding: utf-8 -*-s
import numpy as np
import random
import os    #save the brain and load the brain
import torch
import torch.nn as nn #for neural networks  
import torch.nn.functional as F   #contains functions form submodule
import torch.optim as optim   #for optimizers
import torch.autograd as autograd   #to convert tensors to variables that contain gradients
from torch.autograd import Variable


class Network(nn.Module): #inheriting from module class
    #insize is is number of input neurons
    #actions are left right up down or output neurons
    def __init__(self, input_size, nb_action):#self specifies its a variable of the object
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #connections between neural layers
        self.fc1 = nn.Linear(input_size, 30)#size of hidden layer
        self.fc2 = nn.Linear(30, nb_action) 
    
    def forward(self, state):
        x = F.relu(self.fc1(state)) #activating hidden neurons
        q_values = self.fc2(x) #output nuerons of neural network
        return q_values

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #max number of transition in memory
        self.memory = []
    
    #puts transitions into memory list
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    #takes samples of memory
    def sample(self, batch_size):
        #list = ((1,2,3),(4,5,6)) zip(*list)= ((1,4),(2,3),(5,6)) 
        #structures the variable to contain tensor and gradiant
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #converts to tensor and gradiant

class Dqn(): #Deep Q network
    #num dimension of input state,actions, delay coeffecient
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] #mean of rewards over time
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) #neural network of deep q learning model
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.005)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    #For AI to know which function to select at each time
    def select_action(self, state):
        #setting up probablities for exploring instead of exploitation
        probs = F.softmax(self.model(Variable(state, volatile = True))*50)
         #if T=0, car doesnt have a brain anymmore
        #softmax([1,2,3])=[0.04,0.11,0.85] => softmax([1,2,3]*3) = [0,0.02,0.98]
        action = probs.multinomial()
        return action.data[0,0]
    
    #Training deep nueral network, forward and backward propogation
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #killing fake dimension to get back simple outputs
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)#temporal difference loss
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #appending new state into memory
        #play the new action after reaching the new state
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: #learn from its actions from the last 100 events
            #batch of the state, state+1,reward and action
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) #going to learn form the 100 transitions
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0] #we can see if the mean of the reward is increasing
        return action
    
    def score(self):
		return sum(self.reward_window)/(len(self.reward_window)+1)

    def save(self):
        torch.save({'statedict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict
                    }, 'lastbrain.pth')
        
    def load(self):
        if os.path.isfile('lastbrain.pth'):
            print("loading save file...")
            checkpoint = torch.load('lastbrain.pth')
            self.model.load_state_dict(checkpoint['statedict'])
            self.model.load_state_dict(checkpoint['optimizer'])
            print("done :)")
        else:
            print("no previous save file found")