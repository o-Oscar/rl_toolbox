import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config



class SimpleActor ():
	def __init__ (self, env):
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		self.lstm_size = 256
		
		self.primitive_nb = 8
		
		with tf.name_scope("init_actor"):
			input = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			obs_ph = input
				
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (actor) : no obs range definded. Proceed with caution")
				
			if config.training["use_blindfold"]:
				obs_ph = env.blindfold.action_blindfold(obs_ph)
			
			mean = layers.Dense(512, activation='relu')(obs_ph)
			mean = layers.Dense(256, activation='relu')(mean)
			action = layers.Dense(self.act_dim, activation='sigmoid')(mean)
			self.primitive_model = tf.keras.Model((input, ()), (action, ()), name="primitive")
			self.influence_model = tf.keras.Model((input, ()), (action, ()), name="influence")
			
		#self.model = tf.keras.Model((input, init_state), (action, end_state), name="actor")
		self.model = tf.keras.Model((input, ()), (action, ()), name="actor")
		#self.model.summary()
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))

	def save_primitive (self, path):
		self.primitive_model.save_weights(path.format("actor", "primitive"), overwrite=True)
	def save_influence(self, path):
		self.influence_model.save_weights(path.format("actor", "influence"), overwrite=True)
		
	def load_primitive (self, path):
		self.primitive_model.load_weights(path.format("actor", "primitive"))
	def load_influence (self, path):
		self.influence_model.load_weights(path.format("actor", "influence"))
	
	def lock_primitive (self):
		for layer in self.primitive_model.layers:
			layer.trainable = False
	
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)
		
class RnnActor ():
	def __init__ (self, env):
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		self.lstm_size = 256
		
		self.primitive_nb = 8
		
		with tf.name_scope("init_actor"):
			input = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			obs_ph = input
				
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (actor) : no obs range definded. Proceed with caution")
				
			if config.training["use_blindfold"]:
				obs_ph = blindfold.actor_blindfold(obs_ph)
			
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(obs_ph, initial_state=init_state)
			
			action = layers.Dense(self.act_dim, activation='sigmoid')(lstm)
			"""
			self.primitive_model = tf.keras.Model((input, ()), (action, ()), name="primitive")
			self.influence_model = tf.keras.Model((input, ()), (action, ()), name="influence")
			"""
			self.primitive_model = tf.keras.Model((input, init_state), (action, end_state), name="primitive")
			self.influence_model = tf.keras.Model((input, init_state), (action, end_state), name="influence")
			
		self.model = tf.keras.Model((input, init_state), (action, end_state), name="actor")
		#self.model = tf.keras.Model((input, ()), (action, (action, )), name="actor")
		#self.model.summary()
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape, dtype=np.float32), np.zeros(init_state_shape, dtype=np.float32))

	def save_primitive (self, path):
		self.primitive_model.save_weights(path.format("actor", "primitive"), overwrite=True)
	def save_influence(self, path):
		self.influence_model.save_weights(path.format("actor", "influence"), overwrite=True)
		
	def load_primitive (self, path):
		self.primitive_model.load_weights(path.format("actor", "primitive"))
	def load_influence (self, path):
		self.influence_model.load_weights(path.format("actor", "influence"))
	
	def lock_primitive (self):
		for layer in self.primitive_model.layers:
			layer.trainable = False
	
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)

class MixtureOfExpert ():
	def __init__ (self, env):
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		self.lstm_size = 16
		
		self.primitive_nb = 2
		
		with tf.name_scope("init_actor"):
			input = layers.Input(shape=(None, env.obs_dim))
			#init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			obs_ph = input
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING : no obs range definded. Proceed with caution")
			obs_ph = env.blindfold.action_blindfold(obs_ph)
			
			all_primitives = []
			self.all_primitives_models = []
			
			with tf.name_scope("primitive"):
				primitive_com = obs_ph#[:,:,:34]
				#primitive_com = layers.Dense(256, activation='relu')(primitive_com)
				for i in range(self.primitive_nb):
					primitive = layers.Dense(512, activation='relu')(primitive_com)
					primitive = layers.Dense(256, activation='relu')(primitive)
					#primitive = layers.Dense(128, activation='relu')(primitive_com)
					primitive = layers.Dense(self.act_dim, activation='sigmoid')(primitive)
					
					self.all_primitives_models.append(tf.keras.Model(input, primitive, name="primitive_i"))
					
					all_primitives.append(primitive)
				all_primitives = tf.stack(all_primitives, axis=3)
				self.primitive_model = tf.keras.Model((input, ()), (all_primitives, ()), name="primitive")
				#self.primitive_model.summary()
				
			
			with tf.name_scope("influence"):
				influence = obs_ph
				influence = layers.Dense(256, activation='relu')(influence)
				influence = layers.Dense(128, activation='relu')(influence)
				influence = layers.Dense(self.primitive_nb, activation='softmax')(influence)
				influence = tf.expand_dims(influence, axis=2)
				self.influence_model = tf.keras.Model((input, ()), (influence, ()), name="influence")
				#self.influence_model.summary()
			
			#action = all_primitives[0] # tf.reduce_sum(all_primitives*influence, axis=3)
			action = tf.reduce_sum(all_primitives*influence, axis=3)
			
			
		self.model = tf.keras.Model((input, ()), (action, ()), name="actor")
		#self.model.summary()
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))

	def save_primitive (self, path):
		self.primitive_model.save_weights(path.format("actor", "primitives"), overwrite=True)
	def save_influence(self, path):
		self.influence_model.save_weights(path.format("actor", "influence"), overwrite=True)
		
	def load_primitive (self, path):
		self.primitive_model.load_weights(path.format("actor", "primitives"))
	def load_influence (self, path):
		self.influence_model.load_weights(path.format("actor", "influence"))
	def load_specific_primitive (self, path, i):
		self.all_primitives_models[i].load_weights(path.format("actor", "primitive"))
	
	def lock_primitive (self):
		for layer in self.primitive_model.layers:
			layer.trainable = False
		
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)
		

class Actor (SimpleActor): # choose here the type of actor SimpleActor, MixtureOfExpert
	pass
