import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy import signal
import os.path as osp
import os
import datetime
import time
import matplotlib.pyplot as plt
import sys
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tensorboard --logdir=tensorboard --host localhost --port 8088

from models.critic import Critic

def get_rollout (expert, env, prim, rollout_len, expert_action_prob, log_std):
	# --- simulating the environements ---
	current_s = env.reset()
	current_s = np.expand_dims(np.stack(current_s), axis=1)
	current_expert_init_state = expert.get_init_state(env.num_envs)
	current_prim_init_state = prim.get_init_state(env.num_envs)
	
	is_env_done = [False for i in range(env.num_envs)]
	all_s = [[] for i in range(env.num_envs)]
	all_a = [[] for i in range(env.num_envs)]
	all_r = [[] for i in range(env.num_envs)]
	all_masks = [[] for i in range(env.num_envs)]
	
	n_env_done = 0
	t = 0
	
	while t < rollout_len:#config.training["rollout_len"]:
		t += 1
		current_s = np.asarray(current_s, dtype=np.float32)
		
		expert_a, current_expert_init_state = expert.model((current_s, current_expert_init_state))
		prim_a, current_prim_init_state = prim.model((current_s, current_prim_init_state))
		
		expert_a = expert_a.numpy()
		prim_a = prim_a.numpy()
		
		#print(current_a.shape)
		if np.random.random() < expert_action_prob:
			step_a = expert_a
		else:
			step_a = prim_a
		
		step_a = step_a + np.random.normal(size=12).reshape(step_a.shape) * np.exp(log_std)
		current_new_s, current_r, current_done = env.step(step_a)
		
		current_a = prim_a
		
		n_env_done = 0
		
		for i, (s, a, r, done) in enumerate(zip(current_s, current_a, current_r, current_done)):
			all_s[i].append(s[0])
			all_a[i].append(a[0])
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
	all_r = np.asarray(all_r, dtype=np.float32)
	all_masks = np.asarray(all_masks)
	all_masks[:,-1] = np.zeros(all_masks[:,-1].shape)
	all_masks = all_masks.astype(np.float32)
	
	return (all_s, all_a, all_r, all_masks)


class Trainer:
	def __init__ (self, expert, critic=None, tensorboard_log=""):
		
		if not tensorboard_log == "":
			self.writer = tf.summary.create_file_writer(tensorboard_log)
		
		self.model_save_interval = 5
		
		self.expert = expert
		self.critic = critic
		
		self.gamma = 0.95
		self.lam = 0.9
		
		self.debug_interval = 5
		self.log_interval = 1
		
		self.create_learning_struct ()
	
	
	
	# --- graph initialization ---
	
	def create_learning_struct (self):
		self.trainable_variables = []
		models = [self.expert.model]
		if self.critic is not None:
			models.append(self.critic.model)
		for model in models:
			for layer in model.layers:
				if layer.trainable:
					self.trainable_variables = self.trainable_variables + layer.trainable_weights
		
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-5)
	
	def get_weights (self):
		return self.expert.get_weights()
			
	def set_weights (self, weights):
		self.expert.set_weights(weights)
	
	
	
	def compute_loss (self, n_step, do_log, expert_init_state, critic_init_state, obs, prim_action, new_value, old_value, reward, mask, learning_rate = 2.5e-4):
		with tf.name_scope("training"):
			if self.critic is not None:
				with tf.name_scope("critic"):
					actor_clip_range = 0.2 # appellation pour des raisons historiques
					cur_value, _ = self.critic.model((obs, critic_init_state))
					deltavclipped = old_value + tf.clip_by_value(cur_value - old_value, -actor_clip_range, actor_clip_range)
					critic_losses1 = tf.square(cur_value - new_value)
					critic_losses2 = tf.square(deltavclipped - new_value)
					critic_loss = .5 * tf.reduce_mean(tf.multiply(tf.maximum(critic_losses1, critic_losses2), mask))/tf.reduce_mean(mask) # original
					#self.critic_loss = .5 * tf.reduce_mean(tf.multiply(critic_losses1 + critic_losses2, self.mask_ph))/tf.reduce_mean(self.mask_ph) # sum
					#self.critic_loss = .5 * tf.reduce_mean(tf.multiply(critic_losses2, self.mask_ph))/tf.reduce_mean(self.mask_ph) # just normal
		
			
			with tf.name_scope("expert"):
				expert_action = self.expert.model((obs, expert_init_state))[0]
				expert_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(expert_action - prim_action), axis=-1), mask))/tf.reduce_mean(mask)
			
			if self.critic is not None:
				loss = critic_loss + expert_loss
			else:
				loss = expert_loss
			
		if self.writer is not None and do_log:
			with self.writer.as_default():
				
				with tf.name_scope("training"):
					tf.summary.scalar('expert_loss', expert_loss, n_step)
					if self.critic is not None:
						tf.summary.scalar('critic_loss', critic_loss, n_step)
					tf.summary.scalar('mean_ep_len', tf.reduce_mean(tf.reduce_sum(mask, axis=1)), n_step)
				with tf.name_scope("optimized"):
					tf.summary.scalar('mean_rew', tf.reduce_mean(tf.multiply(reward, mask))/tf.reduce_mean(mask), n_step)
		
		
		return loss
	
	@tf.function 
	def train_step (self, n_step, do_log, expert_init_state, critic_init_state, obs, prim_action, new_value, old_value, reward, mask, learning_rate = 2.5e-4):
		with tf.GradientTape() as tape:
			loss = self.compute_loss(n_step = n_step, do_log = do_log,
									expert_init_state = expert_init_state,
									critic_init_state = critic_init_state,
									obs = obs,
									prim_action = prim_action,
									new_value = new_value,
									old_value = old_value,
									reward = reward,
									mask = mask,
									learning_rate = learning_rate)
		max_grad_norm = 0.5
		gradients = tape.gradient(loss, self.trainable_variables)
		if max_grad_norm is not None:
			gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
		grad_and_var = zip(gradients, self.trainable_variables)
		self.optimizer.learning_rate = learning_rate
		self.optimizer.apply_gradients(grad_and_var)
	
	@tf.function
	def calc_value(self, obs, state):
		value, end_state = self.critic.model((obs, state))
		return value
	
	def train_networks (self, n, all_s, all_a, all_r, all_masks, train_step_nb):
		num_envs = all_s.shape[0]
		
		if self.critic is not None:
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
			
			critic_init_state = self.critic.get_init_state(num_envs)
		else:
			all_last_values = 0
			all_new_value = 0
			critic_init_state = 0
		
		# --- training the networks ---
		for i in range(train_step_nb):
			n_step = tf.constant(n, dtype=tf.int64)
			do_log = tf.convert_to_tensor((n%self.log_interval==0 and i == 0), dtype=tf.bool)
			
			self.train_step(n_step = n_step, do_log=do_log, 
								expert_init_state = self.expert.get_init_state(num_envs),
								critic_init_state = critic_init_state,
								obs = all_s,
								prim_action = all_a,
								new_value = all_new_value,
								old_value = all_last_values,
								reward = all_r,
								mask = all_masks,
								learning_rate = 2.5e-4)

		# --- save the model ---
		if (n+1)%self.model_save_interval == 0:
			self.save()
	
	def save (self):
		path = self.expert.save_path# osp.join(self.actor.save_path, "{}")
		print("Model saved at : " + path.replace("\\", "\\\\"))
		self.expert.save(path)
	
			
			
			
			
			
			