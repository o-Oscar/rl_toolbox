import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ObsScaler:
	def __init__ (self, env):
		self.mean = np.zeros(shape=(env.obs_dim,))
		self.std = np.ones(shape=(env.obs_dim,)) * 0.001
		
		self.lamb = 0.99
	
	def update (self, obs):
		"""
		M = obs.max(axis=(0, 1))
		m = obs.min(axis=(0, 1))
		mean_hat = (M+m)/2 #np.mean(obs, axis=(0, 1))
		std_hat = M-m #np.std(obs, axis=(0, 1))
		
		mean_lamb = self.lamb
		std_lamb = self.lamb
		
		#self.mean = mean_lamb * self.mean + (1-mean_lamb) * mean_hat
		#self.std = std_lamb * self.std + (1-std_lamb) * std_hat
		
		#self.std = np.maximum(self.std, std_hat)
		"""
		cur_M = self.mean+self.std
		cur_m = self.mean-self.std
		
		M = obs.max(axis=(0, 1))
		m = obs.min(axis=(0, 1))
		
		new_M = np.maximum(M, cur_M)
		new_m = np.minimum(m, cur_m)
		
		self.mean = (new_M+new_m)/2
		self.std = (new_M-new_m)/2
		
		
	def scale_obs (self, obs):
		return ((obs - self.mean)/(self.std + 1e-7)).astype(np.float32)
	
	def save (self, path):
		np.save(path.format("scaler_mean") + ".npy", self.mean)
		np.save(path.format("scaler_std") + ".npy", self.std)
	
	def load (self, path):
		self.mean = np.load (path.format("scaler_mean") + ".npy")
		self.std = np.load (path.format("scaler_std") + ".npy")
	
	def get_weights (self):
		return (self.mean, self.std)
	
	def set_weights (self, data):
		self.mean, self.std = data

class BaseActor ():
	def __init__ (self, env):
		self.save_path = "will not work"
		self.scaler = ObsScaler (env)
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		
		self.logstd = tf.Variable(np.ones((self.act_dim,))*(-3), dtype=tf.float32, trainable=True)
		
	def get_weights (self):
		return (self.core_model.get_weights(), self.logstd.value(), self.scaler.get_weights())
		
	def set_weights (self, data):
		weights, logstd_value, scaler_data = data
		self.core_model.set_weights(weights)
		self.logstd.assign(logstd_value)
		self.scaler.set_weights(scaler_data)
		
	def save (self, path):
		self.core_model.save_weights(path.format("actor"), overwrite=True)
		self.scaler.save(path)
		
	def load (self, path):
		self.core_model.load_weights(path.format("actor"))
		self.scaler.load(path)

class SimpleActor (BaseActor):
	def __init__ (self, env, first_size=512, secound_size=256, activation='relu'):
		super().__init__(env)
		
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
			
			mean = layers.Dense(first_size, activation=activation)(obs_input)
			mean = layers.Dense(secound_size, activation=activation)(mean)
			
			skip = obs_input
			
			last_layer = layers.Dense(self.act_dim, activation='tanh')
			action = (last_layer(tf.concat((mean, skip), axis=-1))+1)/2
			
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph)[0], ()), name="actor_model")
		self.model.summary()
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))

class MixtureOfExpert (BaseActor):
	def __init__ (self, env, primitives, debug=False):
		super().__init__(env)
		
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
			self.primitives_cpy = []
			for i, prim in enumerate(primitives):
				#prim.name = str(i) + "coucou"
				self.primitives_cpy.append(tf.keras.Model(inputs=prim.core_model.input, outputs=prim.core_model.output, name='primitive_'+str(i)))
			
			for prim in self.primitives_cpy:
				for layer in prim.layers:
					layer.trainable = True
	
			
			primitives_actions = [prim(obs_input)[0] for prim in self.primitives_cpy]
			primitives_actions = tf.stack(primitives_actions, axis=3)
			
			# action
			action = tf.reduce_sum(primitives_actions*influence, axis=3)
			
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph)[0], ()), name="actor_model")
		
		if debug:
			core_inf_model = tf.keras.Model((obs_input, ()), (influence, ()), name="core_inf_model")
			self.inf_model = tf.keras.Model((input, ()), (core_inf_model(obs_ph)[0], ()), name="inf_model")
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))

class oldLSTMActor (BaseActor):
	def __init__ (self, env):
		super().__init__(env)
		self.lstm_size = 128
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(None, env.obs_dim))
			main_init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
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
			init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
			
			#influence = layers.Dense(512, activation='relu')(obs_input)
			influence = obs_input
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True, activation='relu')(influence, initial_state=init_state)
			
			last_layer = layers.Dense(self.act_dim, activation='tanh')
			action = (last_layer(lstm)+1)/2
			
			self.core_model = tf.keras.Model((obs_input, init_state), (action, end_state), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, main_init_state), self.core_model((obs_ph, main_init_state)), name="actor_model")
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
		#last_layer.set_weights([x/100 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape, dtype=np.float32), np.zeros(init_state_shape, dtype=np.float32))

class LSTMActor (BaseActor):
	def __init__ (self, env):
		super().__init__(env)
		self.lstm_size = 128
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(None, env.obs_dim))
			main_init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
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
			init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
			
			skip = obs_input
			
			influence = layers.Dense(128, activation='relu')(obs_input)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(influence, initial_state=init_state)
			conc = tf.concat((lstm, skip), axis=-1)
			conc = lstm
			
			last_layer = layers.Dense(self.act_dim, activation='tanh')
			action = (last_layer(conc)+1)/2
			
			self.core_model = tf.keras.Model((obs_input, init_state), (action, end_state), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, main_init_state), self.core_model((obs_ph, main_init_state)), name="actor_model")
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
		#last_layer.set_weights([x/100 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape, dtype=np.float32), np.zeros(init_state_shape, dtype=np.float32))