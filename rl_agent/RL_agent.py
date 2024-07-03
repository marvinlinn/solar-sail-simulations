#!/usr/bin/env python
# coding: utf-8

# # RL Agent Test
# ---
# 
# ## Imports

# In[1]:


import os
import sys
current = os.getcwd()
if (os.path.basename(current) == 'rl_agent'):
    top_level_dir = os.path.dirname(os.getcwd())
else:
    top_level_dir = current
sys.path.append(os.path.abspath(top_level_dir))
os.chdir(top_level_dir)

from math import pi
import numpy as np
from rl_agent.World import *
from rl_agent.Agent import *

import matplotlib.pyplot as plt

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
print(f'TensorFlow version: {tf.__version__}')


# ## Instantiate Neural Networks for Policy and Q

# So for now I am using pretty uninformed choices for neural network
# architecture just to get this running asap, but we might want to 
# keep the networks small even when we do this for real.

# In[2]:


class Policy(Model):
    min_action = -pi
    max_action = pi
    
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(512, activation='relu', input_shape=(12,))
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(64, activation='sigmoid')
        self.dropout = Dropout(0.1)
        self.mu = Dense(2)
        self.sigma = Dense(2)    

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = tf.math.softplus(sigma)
        sigma = tf.clip_by_value(sigma, 1e-2, 2*3.14)
        return mu, sigma 

# Create an instance of the model
policy = Policy()
inp = np.random.random((12,12))
print(f'input: \n{inp}\n')
mu, sigma = policy(inp)
print(f'mu: \n{mu}\nsigma: \n{sigma}\n')

dists = tfp.distributions.Normal(mu, sigma)

print(f'dists: \n:{dists}\n')

samples = dists.sample()
print(f'samples: \n{samples}\n')

print(f'log probs: {dists.log_prob(samples.numpy())}\n')

policy.summary()

Q = tf.keras.models.Sequential([
  tf.keras.layers.Dense(512, activation='relu', input_shape=(14,)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='sigmoid'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(1)
], name='Q(s,a)')
Q.summary()


# In[3]:


x = np.random.random((10,3))
a = tf.ones((1,3))
with tf.GradientTape() as tape:
    tape.watch(a)
    y = a * x
print(y)
grad = tape.gradient(y, a)
print(f'sum grad: {tf.reduce_sum(x, axis=0)}')
print(f'len: {len(grad)} \n {grad}')


# ## Instantiate World

# In[4]:


world = ParallelTrackNEO(num_sails=8192)


# ## Instantiate Agent

# In[5]:


agent = ParallelAgent(world, policy, Q, learning_rate_policy=0.00001, learning_rate_Q=0.00056)


# ## Train Agent

# In[6]:


policy.load_weights('./checkpoints/policy.weights.h5')
Q.load_weights('./checkpoints/Q.weights.h5')


# In[ ]:


s = np.array((1,2,3,4,5,6,7,8,9,10,11,12))
s = np.expand_dims(s, axis=0)

preds = policy(s)
print(preds)

EPOCHS = 750
EPISODES = 5

def u(epoch, episode):
    progress_remaining = 1 - (epoch * EPISODES + episode) / (EPOCHS * EPISODES)
    return min(progress_remaining / 0.5, 1) * 1.5

i_0 = 55 + 88 + 75 + 23 + 74 + 75 + 75 + 75 + 40 + 75

u_shifted = lambda epoch, episode: u(epoch + i_0, episode)

agent.train(300, EPISODES, EPOCHS-i_0, added_uncertainty=u)

preds = policy(s)
print(preds)
