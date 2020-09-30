import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy import signal
import os.path as osp
import os
import datetime
import time
import matplotlib.pyplot as plt
import io
import h5py
import sys
import shutil

import warehouse
from models.actor import Actor
from models.critic import Critic

import config
# coucou
#tensorboard --logdir=tensorboard --host localhost --port 8088

USE_SYMETRY = config.training['use_symetry']

if USE_SYMETRY:
	import symetry

def start (env):
	# --- initialization ---
	actor = Actor(env)
	#actor.lock_primitive()
	critic = Critic(env)
	trainer = PPO(actor, critic, env)
	
	parameter_nb = 0
	for var in trainer.trainable_variables:
		parameter_nb += np.prod(var.shape)
	print("Tuning {} parameters".format(parameter_nb))

	# --- creating log dirs ---
	exp_name = "default"
	if len(sys.argv) > 1:
		exp_name = sys.argv[1]
	
	save_dir_path = os.path.join("results", exp_name)#datetime.datetime.now().strftime("snoopx_%Y_%m_%d_%Hh%Mm%Ss"))
	
	if os.path.exists(save_dir_path) and os.path.isdir(save_dir_path): # del dir if exists
		shutil.rmtree(save_dir_path)
	
	tensorboard_log = os.path.join(save_dir_path, "tensorboard")
	os.makedirs(tensorboard_log)
	model_path = os.path.join(save_dir_path, "models")
	os.makedirs(model_path)
	
	trainer.set_model_log_path (model_path, 100)
	
	
	# --- actual training ---
	if config.training["use_init_model"]:
		path = config.training["init_model_path"]
		trainer.restore(path)
	
	N = config.training["ppo_epoch_nb"]
	writer = tf.summary.create_file_writer(tensorboard_log)
	trainer.train(N, writer)
	
	trainer.save()
	
	env.close()
	
	
	

class PPO ():
	def __init__ (self, actor, critic, env):
		self.actor = actor
		self.critic = critic
		self.env = env
		
		self.gamma = 0.95
		self.lam = 0.9
		
		self.create_learning_struct ()
		
		self.model_path = None
		self.model_save_interval = 1
		
		self.debug_interval = 5
	
		self.publish_model ()
	
	# --- initialization ---
	
	@tf.function
	def create_neglogp (self, full_state, x):
		action, new_state = self.actor.model(full_state)
		return self.create_neglogp_act(action, x)
	
	def create_neglogp_act (self, act, x):
		return 0.5 * tf.reduce_sum(tf.square((x - act) / self.std), axis=-1) \
			   + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], dtype=tf.float32) \
			   + tf.reduce_sum(self.logstd, axis=-1)
	
	def create_learning_struct (self):
		
		self.trainable_variables = []
		for model in [self.actor.model, self.critic.model]:
			for layer in model.layers:
				if layer.trainable:
					self.trainable_variables = self.trainable_variables + layer.trainable_weights
		
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-5)
		
		# randomize actor (exploration)
		with tf.name_scope("randomizer"):
			with tf.name_scope("stochastic"):
				init_logstd = -3 # dogo 
				#init_logstd = -2 # cartpole 
				self.logstd = tf.Variable(np.ones((self.actor.act_dim,))*init_logstd, dtype=tf.float32, trainable=False) # ---------- is the std trainable ???
				self.std = tf.exp(self.logstd)
	
	@tf.function
	def step (self, obs, state, deterministic=False):
		if deterministic:
			
			action, final_state = self.actor.model((obs, state))
			neglog = np.zeros(tf.shape(action).numpy()[:-1])
		else:
			action, final_state = self.actor.model((obs, state))
			if hasattr(self.env, 'act_a'):
				action += self.std * tf.random.normal(tf.shape(action))# * self.env.act_a
			else:
				action += self.std * tf.random.normal(tf.shape(action))
			neglog = self.create_neglogp ((obs, state), action)
		return action, final_state, neglog
	
	@tf.function
	def calc_value(self, obs, state):
		value, end_state = self.critic.model((obs, state))
		return value
	
	def compute_loss (self, n_step, do_log, actor_init_state, critic_init_state, obs, action, advantage, new_value, reward, old_neglog, old_value, mask, learning_rate = 2.5e-4, actor_clip_range = 0.2, critic_clip_range = 1):
		with tf.name_scope("training"):
			with tf.name_scope("critic"):
				cur_value, _ = self.critic.model((obs, critic_init_state))
				deltavclipped = old_value + tf.clip_by_value(cur_value - old_value, -actor_clip_range, actor_clip_range)
				critic_losses1 = tf.square(cur_value - new_value)
				critic_losses2 = tf.square(deltavclipped - new_value)
				critic_loss = .5 * tf.reduce_mean(tf.multiply(tf.maximum(critic_losses1, critic_losses2), mask))/tf.reduce_mean(mask) # original
				#self.critic_loss = .5 * tf.reduce_mean(tf.multiply(critic_losses1 + critic_losses2, self.mask_ph))/tf.reduce_mean(self.mask_ph) # sum
				#self.critic_loss = .5 * tf.reduce_mean(tf.multiply(critic_losses2, self.mask_ph))/tf.reduce_mean(self.mask_ph) # just normal
		
		
			with tf.name_scope("actor"):
				with tf.name_scope("train_neglog"):
					train_neglog = self.create_neglogp((obs, actor_init_state), action)
				with tf.name_scope("ratio"):
					ratio = tf.exp(old_neglog - train_neglog)
				with tf.name_scope("loss"):
					actor_loss1 = -advantage * ratio
					actor_loss2 = -advantage * tf.clip_by_value(ratio, 1.0 - actor_clip_range, 1.0 + actor_clip_range)
					actor_loss = tf.reduce_mean(tf.multiply(tf.maximum(actor_loss1, actor_loss2), mask))/tf.reduce_mean(mask)
				
				with tf.name_scope("probe"):
					with tf.name_scope("approxkl"):
						approxkl = .5 * tf.reduce_mean(tf.square(train_neglog - old_neglog))
					with tf.name_scope("clipfrac"):
						clipfrac = tf.reduce_mean(tf.multiply(tf.cast(tf.greater(tf.abs(ratio - 1.0), actor_clip_range), tf.float32), mask))/tf.reduce_mean(mask)
			if USE_SYMETRY:
				with tf.name_scope("symetry"):
					symetry_loss = symetry.loss(self.actor, obs, actor_init_state, mask)
			# l'entropie ne sert a rien pk ell est constante
			# le coefficient devant le critic loss ne sert a rien tant que le critic et l'acteur ne partagent aucun parametre.
			with tf.name_scope("loss"):
				#self.loss = self.actor_loss - self.entropy * 0.01 + self.critic_loss * 0.5
				loss = actor_loss + critic_loss * 0.5
				if USE_SYMETRY:
					loss = loss + symetry_loss * 4
		
		if self.writer is not None and do_log:
			with self.writer.as_default():
				
				with tf.name_scope("training"):
					tf.summary.scalar('actor_loss', actor_loss, n_step)
					tf.summary.scalar('critic_loss', critic_loss, n_step)
					tf.summary.scalar('mean_ep_len', tf.reduce_mean(tf.reduce_sum(mask, axis=1)), n_step)
				with tf.name_scope("optimized"):
					if USE_SYMETRY:
						tf.summary.scalar('symetry_loss', symetry_loss, n_step)
					tf.summary.scalar('discounted_rewards', tf.reduce_mean(tf.multiply(old_value + advantage, mask))/tf.reduce_mean(mask), n_step)
					tf.summary.scalar('mean_rew', tf.reduce_mean(tf.multiply(reward, mask))/tf.reduce_mean(mask), n_step)
		
		
		return loss
	
	@tf.function
	def train_step (self, n_step, do_log, actor_init_state, critic_init_state, obs, action, advantage, new_value, reward, old_neglog, old_value, mask, learning_rate = 2.5e-4, actor_clip_range = 0.2, critic_clip_range = 1):
		with tf.GradientTape() as tape:
			loss = self.compute_loss(n_step = n_step, do_log = do_log,
									actor_init_state = actor_init_state,
									critic_init_state = critic_init_state,
									obs = obs,
									action = action,
									advantage = advantage,
									new_value = new_value,
									reward = reward,
									old_neglog = old_neglog,
									old_value = old_value,
									mask = mask,
									learning_rate = learning_rate,
									actor_clip_range = actor_clip_range,
									critic_clip_range = critic_clip_range)
		max_grad_norm = 0.5
		gradients = tape.gradient(loss, self.trainable_variables)
		if max_grad_norm is not None:
			gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
		grad_and_var = zip(gradients, self.trainable_variables)
		self.optimizer.learning_rate = learning_rate
		self.optimizer.apply_gradients(grad_and_var)
	
	def train (self, total_timesteps, writer=None, log_interval=1):
		self.writer = writer
		
		batch_size = 100 # max ep len
		batch_size_option = [100] # [30, 50, 70, 100, 150, 200, 300]
		
		last_debug_time = 0
	
		print("Starting learning")
		n = 0
		end_requested = False
		start_time = time.time()
		
		t = 0
		
		while n < total_timesteps and not end_requested:
			n += 1
			
			# --- debug ---
			if n%self.debug_interval == 0:
				print("step : {}".format(n))
				if n > 0:
					print("fps : " + str(self.env.num_envs*t*self.debug_interval/(time.time()-last_debug_time))) # <---- TODO : change self.env.num_envs*t to the actual number of processed frames
					print("batch_size (t) : " + str(t), flush=True)
					
				last_debug_time = time.time()
			"""
			# --- simulating the environements ---
			self.env.set_epoch(n)
			current_s = self.env.reset()
			current_s = np.expand_dims(np.stack(current_s), axis=1)
			current_actor_init_state = self.actor.get_init_state(self.env.num_envs)
			
			is_env_done = [False for i in range(self.env.num_envs)]
			all_s = [[] for i in range(self.env.num_envs)]
			all_a = [[] for i in range(self.env.num_envs)]
			all_neglog = [[] for i in range(self.env.num_envs)]
			all_r = [[] for i in range(self.env.num_envs)]
			all_masks = [[] for i in range(self.env.num_envs)]
			
			n_env_done = 0
			t = 0
			
			while t < batch_size and not (n_env_done >= self.env.num_envs*0.75 and t in batch_size_option):
				t += 1
				current_s = np.asarray(current_s, dtype=np.float32)
				current_a, current_actor_init_state, current_neglog = self.step (current_s, current_actor_init_state)
				current_a = current_a.numpy()
				current_neglog = current_neglog.numpy()
				#print(current_a.shape)
				current_new_s, current_r, current_done = self.env.step(current_a)
				
				n_env_done = 0
				
				for i, (s, a, neglog, r, done) in enumerate(zip(current_s, current_a, current_neglog, current_r, current_done)):
					all_s[i].append(s[0])
					all_a[i].append(a[0])
					all_neglog[i].append(neglog[0])
					if not is_env_done[i]:
						all_r[i].append(r)
						all_masks[i].append(1)
						is_env_done[i] = done
					else:
						all_r[i].append(r)
						all_masks[i].append(0)
						n_env_done += 1
				
				current_s = current_new_s
				current_s = np.expand_dims(np.stack(current_s), axis=1)
				
				
			
			# --- reshaping the logs ---
			all_s = np.asarray(all_s, dtype=np.float32)
			all_a = np.asarray(all_a, dtype=np.float32)
			all_neglog = np.asarray(all_neglog, dtype=np.float32)
			all_r = np.asarray(all_r, dtype=np.float32)
			all_masks = np.asarray(all_masks)
			all_masks[:,-1] = np.zeros(all_masks[:,-1].shape)
			all_masks = all_masks.astype(np.float32)
			
			num_envs = self.env.num_envs
			"""
			
			"""
			rollouts = np.asarray(warehouse.load(warehouse.ROLLOUT), dtype=np.float32)
			if len(rollouts.shape) < 2:
				time.sleep(0.03)
			if len(rollouts.shape) >= 2:
			
				all_s = rollouts[0]
				all_a = rollouts[1]
				all_neglog = rollouts[2]
				all_r = rollouts[3]
				all_masks = rollouts[4]
				#all_masks[:,-1] = np.zeros(all_masks[:,-1].shape, dtype=np.float32)
			"""
			
			all_s = warehouse.load(warehouse.ROLLOUT_STATE)
			all_a = warehouse.load(warehouse.ROLLOUT_ACTION)
			all_neglog = warehouse.load(warehouse.ROLLOUT_NEGLOG)
			all_r = warehouse.load(warehouse.ROLLOUT_REW)
			all_masks = warehouse.load(warehouse.ROLLOUT_MASK)
			
			while len(all_s.shape) < 2:
				time.sleep(0.03)
				
				all_s = warehouse.load(warehouse.ROLLOUT_STATE)
				all_a = warehouse.load(warehouse.ROLLOUT_ACTION)
				all_neglog = warehouse.load(warehouse.ROLLOUT_NEGLOG)
				all_r = warehouse.load(warehouse.ROLLOUT_REW)
				all_masks = warehouse.load(warehouse.ROLLOUT_MASK)
				
				
			if len(all_s.shape) >= 2:
		
			

			
				num_envs = all_s.shape[0]
				print("Loaded rollout nb : ", num_envs)
				
				
				# --- calculating gae ---
				val = self.calc_value(all_s, self.critic.get_init_state(num_envs)).numpy()
				all_last_values = val * all_masks + all_r * (1-all_masks) / (1-self.gamma)
				
				
				all_better_value = np.array(all_r, copy=True)
				all_better_value[:,:-1] += self.gamma*all_last_values[:,1:]
				all_better_value[:,-1] = all_last_values[:,-1]
				all_deltas = all_better_value - all_last_values
				
				all_gae = np.flip(signal.lfilter([1], [1, -self.gamma*self.lam], np.flip(all_deltas, axis=1)), axis=1)
				all_new_value = all_last_values + all_gae
				all_gae = (all_gae - all_gae.mean()) / (all_gae.std() + 1e-8)
				
				all_last_values = all_last_values.astype(np.float32)
				all_deltas = all_deltas.astype(np.float32)
				all_gae = all_gae.astype(np.float32)
				all_new_value = all_new_value.astype(np.float32)
				
				# --- training the networks ---
				for i in range(config.training["ppo_train_step_nb"]):
					n_step = tf.constant((n-1), dtype=tf.int64)
					self.train_step(n_step = n_step, do_log=(n%log_interval==0 and i == 0), 
										actor_init_state = self.actor.get_init_state(num_envs),
										critic_init_state = self.critic.get_init_state(num_envs),
										obs = all_s,
										action = all_a,
										advantage = all_gae,
										new_value = all_new_value,
										reward = all_r,
										old_neglog = all_neglog,
										old_value = all_last_values,
										mask = all_masks,
										learning_rate = 2.5e-4,
										actor_clip_range = 0.2,
										critic_clip_range = 1)
										
					# --- sending the network to the rollout workers ---
					self.publish_model ()
					
					
				self.writer.flush()
		
		
				# --- save the model ---
				if self.model_path is not None and n%self.model_save_interval == 0:
					self.save()
				
				# --- end training early if needed ---
				#end_requested = (os.path.getmtime("stop_file.txt") > start_time)
			
		warehouse.close_warehouse()
			
			
	def set_model_log_path (self, model_path, interval):
		self.model_path = model_path
		self.model_save_interval = interval
	
	def publish_model (self):
		bio = io.BytesIO()
		with h5py.File(bio, "w") as f:
			self.actor.model.save(f)
		model_data = bio.getvalue()
		
		warehouse.save(warehouse.ACT_NET, model_data)
	
	def save (self):
		path = osp.join(self.model_path, "{}_{}")
		print("Model saved at : " + path.replace("\\", "\\\\"))
		self.actor.save_primitive(path)
		self.actor.save_influence(path)
		self.critic.model.save_weights(path.format("critic", "model"), overwrite=True)
	
	def restore (self, path):
		self.actor.load_influence(path)
		self.actor.load_primitive(path)
		self.critic.model.load_weights(path.format("critic", "model"))
	