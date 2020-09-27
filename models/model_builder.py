import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers
"""
class Actor ():
	def __init__ (self, env):
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		self.lstm_size = 16
		with tf.name_scope("init_actor"):
			obs_ph = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]

			mean = obs_ph
			mean = layers.Dense(512, activation='relu')(mean)
			mean = layers.Dense(256, activation='relu')(mean)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			#mean = layers.concatenate([mean, lstm])
			mean = layers.Dense(self.act_dim, activation='tanh')(mean)
			
		self.model = tf.keras.Model((obs_ph, init_state), (mean, [var for var in init_state]), name="actor")
		
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))
		"""
		
class Actor ():
	def __init__ (self, env):
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		self.lstm_size = 16
		with tf.name_scope("init_actor"):
			obs_ph = layers.Input(shape=(None, env.obs_dim))
			#init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			mean = obs_ph
			
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				mean = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING : no obs range definded. Proceed with caution")
			
			mean = layers.Dense(512, activation='relu')(mean)
			#mean = layers.Dense(512, activation='relu')(mean)
			mean = layers.Dense(256, activation='relu')(mean)
			mean = layers.Dense(self.act_dim, activation='sigmoid')(mean)
			"""
			if hasattr(env, 'act_a') and  hasattr(env, 'act_b'):
				mean = mean*env.act_a + env.act_b
				print(env.act_b)
			else:
				print("WARNING : no action range definded. Proceed with caution")
			"""
		self.model = tf.keras.Model((obs_ph, ()), (mean, ()), name="actor")
		
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))

class Critic ():
	def __init__ (self, env):
		self.obs_dim = env.obs_dim
		self.lstm_size = 16
		with tf.name_scope("init_critic"):
			obs_ph = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			mean = obs_ph
			
			if hasattr(env, 'obs_mean'):
				mean = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING : no obs range definded. Proceed with caution")
			
			mean = layers.Dense(512, activation='relu')(mean)
			#mean = layers.Dense(512, activation='relu')(mean)
			mean = layers.Dense(256, activation='relu')(mean)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			#mean = layers.concatenate([mean, lstm])
			mean = tf.squeeze(layers.Dense(1, activation='linear')(mean), axis=[2])
	
		#self.model = tf.keras.Model((obs_ph, ()), (mean, ()), name="critic")
		self.model = tf.keras.Model((obs_ph, init_state), (mean, end_state), name="critic")
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))
"""
class Critic ():
	def __init__ (self, env):
		self.obs_dim = env.obs_dim
		self.lstm_size = 16
		with tf.name_scope("init_critic"):
			obs_ph = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			mean = obs_ph
			mean = layers.Dense(512, activation='relu')(mean)
			mean = layers.Dense(256, activation='relu')(mean)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			#mean = layers.concatenate([mean, lstm])
			mean = tf.squeeze(layers.Dense(1, activation='linear')(mean), axis=[2])
	
		self.model = tf.keras.Model((obs_ph, init_state), (mean, end_state), name="critic")
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))



"""

