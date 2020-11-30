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
from ppo_distillation import PPO
import warehouse

class TrainDistillationNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		
		primitive_nb = int(self.data['primitive_nb_prop'])
		envs = []
		primitives = []
		for i in range(primitive_nb):
			envs.append(input_dict['Env_'+str(i)][0])
			primitives.append(input_dict['Primitive_'+str(i)][0])
		
		env = input_dict['Env'][0]
		actor = input_dict['Expert'][0]
		log_std = float(self.data['log_std_prop'])
		tau = float(self.data['tau_prop'])
		
		USE_ADR = False #hasattr(env, 'adr')
		
		if self.mpi_role == 'main':
			tensorboard_path = os.path.join(save_path['tensorboard'], self.data['tensorboard_name_prop'])
			os.makedirs(tensorboard_path)
			
			trainer = PPO(env, actor, tensorboard_path, init_log_std=log_std)
			trainer.model_save_interval = int(self.data['model_save_interval_prop'])
			train_step_nb = int(self.data['train_step_nb_prop'])
			if "Critic" in input_dict:
				trainer.critic = input_dict['Critic'][0]
			trainer.tau = tau
			
			start_time = time.time()
			desired_rollout_nb = int(self.data['rollout_nb_prop'])
			
			prim_weights = [prim.get_weights() for prim in primitives]
			
			for n in range(int(self.data['epoch_nb_prop'])):
				# send the network weights
				# and get the latest rollouts
				msg = {"node":proc_num, "weights" : trainer.get_weights(), "primitive" : prim_weights, "rollout_nb":desired_rollout_nb, "request" : ["s", "a", "prim_a", "r", "neglog", "mask", "dumped", "adr"]}
				data = warehouse.send(msg)
				all_s = data["s"]
				all_a = data["a"]
				all_prim_a = data["prim_a"]
				all_r = data["r"]
				all_neglog = data["neglog"]
				all_masks = data["mask"]
				dumped_rollout_nb = data["dumped"]
				if USE_ADR:
					env.adr.update(data["adr"])
					env.adr.log()
				
				# update the network weights
				trainer.train_networks(n, all_s, all_a, all_prim_a, all_r, all_neglog, all_masks, train_step_nb)
				
				#debug
				n_rollouts = all_s.shape[0]
				rollout_len = all_s.shape[1]
				print("Epoch {} :".format(n), flush=True)
				print("Loaded {} rollouts for training while dumping {}.".format(n_rollouts, dumped_rollout_nb), flush=True)
				dt = time.time() - start_time
				start_time = time.time()
				if dt > 0:
					print("fps : {}".format(n_rollouts*rollout_len/dt), flush=True)
				print("mean_rew : {}".format(np.sum(all_r * all_masks)/np.sum(all_masks)), flush=True)
				
				env.adr.save()
			env.adr.save()
		elif self.mpi_role == 'worker':
			trainer = PPO(env, actor, init_log_std=log_std)
			
			rollout_len = int(self.data['rollout_len_prop'])
			#data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
			msg = {"request" : ["weights", "primitive", "node"]}
			data = warehouse.send(msg)
			
			while proc_num > data["node"]:
				time.sleep(0.3)
				data = warehouse.send(msg)
			
			for prim, prim_weight in zip(primitives, data["primitive"]):
				prim.set_weights(prim_weight)
			
			k = 0
			while proc_num == data["node"]:
				k += 1
				test_adr = USE_ADR and np.random.random() < float(self.data['adr_prob_prop'])
				
				id = int(np.random.random()*len(envs))
				cur_env = envs[id]
				cur_prim = primitives[id]
				
				cur_env.test_adr = test_adr
				
				#print(data["weights"][0], flush=True)
				actor.set_weights (data["weights"])
				
				if test_adr:
					# simulate rollout
					all_s, all_a, all_r, all_neglog, all_mask = get_rollout(env.adr_rollout_len)
					
					msg = {"node":proc_num, 
							"adr" : env.adr.get_msg(),
							"request" : ["weights", "adr", "node"]}
				else:
					current_s = cur_env.reset()
					current_s = np.expand_dims(np.stack(current_s), axis=1)
					current_actor_init_state = cur_prim.get_init_state(cur_env.num_envs)
					
					for i in range(10):
						current_s = np.asarray(current_s, dtype=np.float32)
						current_a, current_actor_init_state = cur_prim.model ((current_s, current_actor_init_state))
						current_a = current_a.numpy()
						
						current_new_s, current_r, current_done = cur_env.step(current_a)
						
						current_s = current_new_s
						current_s = np.expand_dims(np.stack(current_s), axis=1)
					
					trainer.env = cur_env
					
					# simulate rollout
					help_fac = max(min(1-k/1000, 1), 0) * 0.5
					all_s, all_a, all_prim_a, all_r, all_neglog, all_mask = get_rollout(trainer, cur_prim, rollout_len, current_s=current_s, help_fac=help_fac)
					
					# send rollout back to warehouse
					# and get network weights and update actor
					msg = {"node":proc_num, 
							"s" : all_s,
							"a" : all_a,
							"prim_a" : all_prim_a,
							"r" : all_r,
							"neglog" : all_neglog,
							"mask" : all_mask,
							"request" : ["weights", "adr", "node"]}
					
				data = warehouse.send(msg)
				
				if USE_ADR:
					env.adr.update(data["adr"])
		
		
		output_dict['Expert'] = trainer.actor
		output_dict['Critic'] = trainer.critic


def get_rollout (trainer, cur_prim, rollout_len, current_s=None, help_fac=0):
	# --- simulating the environements ---
	if current_s is None:
		current_s = trainer.env.reset()
		current_s = np.expand_dims(np.stack(current_s), axis=1)
	current_actor_init_state = trainer.actor.get_init_state(trainer.env.num_envs)
	current_prim_init_state = cur_prim.get_init_state(trainer.env.num_envs)
	
	is_env_done = [False for i in range(trainer.env.num_envs)]
	all_s = [[] for i in range(trainer.env.num_envs)]
	all_a = [[] for i in range(trainer.env.num_envs)]
	all_prim_a = [[] for i in range(trainer.env.num_envs)]
	all_neglog = [[] for i in range(trainer.env.num_envs)]
	all_r = [[] for i in range(trainer.env.num_envs)]
	all_masks = [[] for i in range(trainer.env.num_envs)]
	
	n_env_done = 0
	t = 0
	
	while t < rollout_len:#config.training["rollout_len"]:
		t += 1
		current_s = np.asarray(current_s, dtype=np.float32)
		current_a, current_actor_init_state, current_neglog = trainer.step (current_s, current_actor_init_state)
		prim_a, current_prim_init_state = cur_prim.model ((current_s, current_prim_init_state))
		current_a = current_a.numpy()
		current_neglog = current_neglog.numpy()
		#print(current_a.shape)
		current_new_s, current_r, current_done = trainer.env.step(current_a)
		
		n_env_done = 0
		
		for i, (s, a, p_a, neglog, r, done) in enumerate(zip(current_s, current_a, prim_a, current_neglog, current_r, current_done)):
			all_s[i].append(s[0])
			all_a[i].append(a[0])
			all_prim_a[i].append(p_a[0])
			all_neglog[i].append(neglog[0])
			all_r[i].append(r)# + (np.exp(-np.sum(np.square(a - p_a))/np.exp(trainer.init_log_std)) - 1) * help_fac)
			if not is_env_done[i]:
				all_masks[i].append(1)
				is_env_done[i] = done
			else:
				all_masks[i].append(0)
				n_env_done += 1
		
		current_s = current_new_s
		current_s = np.expand_dims(np.stack(current_s), axis=1)
		
		
	
	# --- reshaping the logs ---
	all_s = np.asarray(all_s, dtype=np.float32)
	all_a = np.asarray(all_a, dtype=np.float32)
	all_prim_a = np.asarray(all_prim_a, dtype=np.float32)
	all_neglog = np.asarray(all_neglog, dtype=np.float32)
	all_r = np.asarray(all_r, dtype=np.float32)
	all_masks = np.asarray(all_masks)
	all_masks[:,-1] = np.zeros(all_masks[:,-1].shape)
	all_masks = all_masks.astype(np.float32)
	
	return (all_s, all_a, all_prim_a, all_r, all_neglog, all_masks)


