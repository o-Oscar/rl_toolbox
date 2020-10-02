import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config

class BaseActor ():
	def __init__ (self):
		self.save_path = "will not work"

	def get_weights (self):
		return self.core_model.get_weights()
		
	def set_weights (self, weights):
		self.core_model.set_weights(weights)
		
	def save (self, path):
		self.core_model.save_weights(path.format("actor"), overwrite=True)
		
	def load (self, path):
		self.core_model.load_weights(path.format("actor"))

class SimpleActor (BaseActor):
	def __init__ (self, env):
		super().__init__()
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(None, env.obs_dim))
			obs_ph = input
			
			# scaling the inputs
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (actor) : no obs range definded. Proceed with caution")
				
			# using the optional blindfold
			if hasattr(env, 'blindfold'):
				obs_ph = env.blindfold.action_blindfold(obs_ph)
			else:
				print("WARNING (actor) : no blindfold used")
			
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(None, obs_ph.shape[-1]))
			
			mean = layers.Dense(512, activation='relu')(obs_input)
			mean = layers.Dense(256, activation='relu')(mean)
			action = layers.Dense(self.act_dim, activation='sigmoid')(mean)
			
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph), ()), name="actor_model")
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))

class MixtureOfExpert (BaseActor):
	def __init__ (self, env, primitives):
		super().__init__()
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		
		#primitives = primitives[:1]
		self.primitive_nb = len(primitives)
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(None, env.obs_dim))
			obs_ph = input
			
			# scaling the inputs
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (actor) : no obs range definded. Proceed with caution")
				
			# using the optional blindfold
			if hasattr(env, 'blindfold'):
				obs_ph = env.blindfold.action_blindfold(obs_ph)
			else:
				print("WARNING (actor) : no blindfold used")
			
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(None, obs_ph.shape[-1]))
			
			# influence
			influence = layers.Dense(512, activation='relu')(obs_input)
			influence = layers.Dense(256, activation='relu')(influence)
			influence = layers.Dense(self.primitive_nb, activation='softmax')(influence)
			influence = tf.expand_dims(influence, axis=2)
			
			# primitives
			primitives_cpy = []
			for i, prim in enumerate(primitives):
				#prim.name = str(i) + "coucou"
				primitives_cpy.append(tf.keras.Model(inputs=prim.core_model.input, outputs=prim.core_model.output, name='primitive_'+str(i)))
			
			primitives_actions = [prim(obs_input)[0] for prim in primitives_cpy]
			primitives_actions = tf.stack(primitives_actions, axis=3)
			
			# action
			action = tf.reduce_sum(primitives_actions*influence, axis=3)
			
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph), ()), name="actor_model")
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))

"""
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
"""