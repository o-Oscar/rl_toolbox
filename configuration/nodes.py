
import os

from models.actor import SimpleActor

class SimpleActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		first_size = int(self.data['first_size_prop'])
		secound_size = int(self.data['secound_size_prop'])
		activation = self.data['activation_prop']
		actor = SimpleActor (env, first_size, secound_size, activation)
		output_dict['Actor'] = actor
		
		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)

from models.actor import MixtureOfExpert

class MixActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		primitives = input_dict['Primitive']
		actor = MixtureOfExpert (env, primitives)
		output_dict['Actor'] = actor

		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)
		
from models.actor import LSTMActor

class LSTMActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		actor = LSTMActor (env)
		output_dict['Actor'] = actor
		
		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)
			
			
			
class LoadActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		actor = input_dict['Actor'][0]
		output_dict['Actor'] = actor

		if self.mpi_role == 'main':
			path = self.data['model_path_prop']
			actor.load(path + "/{}")
			
			#save the actor
			actor.save(actor.save_path)

from models.critic import Critic

class LoadCriticNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		path = self.data['model_path_prop'] + "/{}"
		critic = Critic (env)
		if self.mpi_role == 'main':
			critic.model.load_weights(path.format("critic"))
		output_dict['Critic'] = critic

# free the whole net
class FreePrimitiveNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		actor = input_dict['Actor'][0]
		for layer in actor.model.layers:
			layer.trainable = (self.data['free_prop'] == "1")
		output_dict['Actor'] = actor

		if self.mpi_role == 'main':
			
			#save the actor
			actor.save(actor.save_path)


from environments import simple_env

class SimpleEnvNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		env = simple_env.SimpleEnv()
		output_dict['Env'] = env

from environments import cartpole

class CartPoleNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		env = cartpole.CartPoleEnv()
		env.mode = int(self.data['mode_prop'])
		output_dict['Env'] = env

from environments import dog_env

class DogEnvNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		env = dog_env.DogEnv()
		if (self.data['full_parkour_prop'] == "1") == (self.data['simple_walk_prop'] == "1"):
			raise NameError('dog env not properly set up')
		#env.only_forward = self.data['simple_walk_prop'] == "1"
		#env.has_rand_act_delta = self.data['rand_delta_prop'] == "1"
		#env.carthesian_act = self.data['carthesian_act_prop'] == "1"
		env.training_mode = int(self.data['mode_prop'])
		env.rewards[6].a *= 0 if self.data['base_rot_rew_prop'] == "0" else 1
		output_dict['Env'] = env

from ppo import PPO
import warehouse
import time
import numpy as np

class TrainPPONode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
	
		env = input_dict['Environment'][0]
		actor = input_dict['Actor'][0]
		log_std = float(self.data['log_std_prop'])
		
		USE_ADR = hasattr(env, 'adr')
		
		if self.mpi_role == 'main':
			tensorboard_path = os.path.join(save_path['tensorboard'], self.data['tensorboard_name_prop'])
			os.makedirs(tensorboard_path)
			
			trainer = PPO(env, actor, tensorboard_path, init_log_std=log_std)
			trainer.model_save_interval = int(self.data['model_save_interval_prop'])
			train_step_nb = int(self.data['train_step_nb_prop'])
			if "Critic" in input_dict:
				trainer.critic = input_dict['Critic'][0]
			
			start_time = time.time()
			desired_rollout_nb = int(self.data['rollout_nb_prop'])
			
			for n in range(int(self.data['epoch_nb_prop'])):
				# send the network weights
				# and get the latest rollouts
				req = ["s", "a", "r", "neglog", "mask", "dumped", "adr"]
				msg = {"node":proc_num, "weights" : trainer.get_weights(), "rollout_nb":desired_rollout_nb, "request" : req}
				data = warehouse.send(msg)
				all_s = data["s"]
				all_a = data["a"]
				all_r = data["r"]
				all_neglog = data["neglog"]
				all_masks = data["mask"]
				dumped_rollout_nb = data["dumped"]
				if USE_ADR:
					env.adr.update(data["adr"])
					env.adr.log()
				
				# update the network weights
				all_last_values, all_gae, all_new_value = trainer.calc_gae(all_s, all_r, all_masks)
				trainer.train_networks(n, all_s, all_a, all_r, all_neglog, all_masks, train_step_nb, all_last_values, all_gae, all_new_value)
				
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
				
				if USE_ADR:
					env.adr.save()
					
		elif self.mpi_role == 'worker':
			trainer = PPO(env, actor, init_log_std=log_std)
			rollout_len = int(self.data['rollout_len_prop'])
			#data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
			msg = {"request" : ["weights", "node"]}
			data = warehouse.send(msg)
			
			while proc_num > data["node"]:
				time.sleep(0.3)
				data = warehouse.send(msg)
			
			while proc_num == data["node"]:
				test_adr = USE_ADR and np.random.random() < float(self.data['adr_prob_prop'])
				
				env.test_adr = test_adr
				
				#print(data["weights"][0], flush=True)
				trainer.set_weights (data["weights"])
				
				if test_adr:
					# simulate rollout
					all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(env.adr_rollout_len)
					
					msg = {"node":proc_num, 
							"adr" : env.adr.get_msg(),
							"request" : ["weights", "adr", "node"]}
				else:
					# simulate rollout
					all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(rollout_len)
					
					# send rollout back to warehouse
					# and get network weights and update actor
					msg = {"node":proc_num, 
							"s" : all_s,
							"a" : all_a,
							"r" : all_r,
							"neglog" : all_neglog,
							"mask" : all_mask,
							"request" : ["weights", "adr", "node"]}
					
				data = warehouse.send(msg)
				
				if USE_ADR:
					env.adr.update(data["adr"])
		
		
		output_dict['Trained actor'] = trainer.actor
		output_dict['Critic'] = trainer.critic

import distillation
import new_distillation

class TrainDistillationNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		
		if "Env" in input_dict:
			env = input_dict['Env'][0]
			critic = Critic (env)
			print("training critic")
		else:
			critic = None
		
		primitive_nb = int(self.data['primitive_nb_prop'])
		envs = []
		primitives = []
		for i in range(primitive_nb):
			envs.append(input_dict['Env_'+str(i)][0])
			primitives.append(input_dict['Primitive_'+str(i)][0])
		expert = input_dict['Expert'][0]
		
		if self.mpi_role == 'main':
			tensorboard_path = os.path.join(save_path['tensorboard'], self.data['tensorboard_name_prop'])
			os.makedirs(tensorboard_path)
			
			trainer = distillation.Trainer(expert, critic, tensorboard_path)
			trainer.model_save_interval = int(self.data['model_save_interval_prop'])
			train_step_nb = int(self.data['train_step_nb_prop'])
			
			start_time = time.time()
			desired_rollout_nb = int(self.data['rollout_nb_prop'])
			
			prim_weights = [prim.get_weights() for prim in primitives]
			
			for n in range(int(self.data['epoch_nb_prop'])):
				# send the network weights
				# and get the latest rollouts
				msg = {"node":proc_num, "weights" : trainer.get_weights(), "primitive" : prim_weights, "rollout_nb":desired_rollout_nb, "request" : ["s", "a", "r", "mask", "dumped"]}
				data = warehouse.send(msg)
				all_s = data["s"]
				all_a = data["a"]
				all_r = data["r"]
				all_masks = data["mask"]
				dumped_rollout_nb = data["dumped"]
				
				# update the network weights
				trainer.train_networks(n, all_s, all_a, all_r, all_masks, train_step_nb)
				
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
				
			output_dict['Expert'] = trainer.expert
			output_dict['Critic'] = trainer.critic
			
		elif self.mpi_role == 'worker':
			rollout_len = int(self.data['rollout_len_prop'])
			expert_action_prob = float(self.data["expert_action_prob_prop"])
			log_std = float(self.data["log_std_prop"])
			
			#data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
			msg = {"request" : ["weights", "primitive", "node"]}
			data = warehouse.send(msg)
			
			while proc_num > data["node"]:
				time.sleep(0.3)
				data = warehouse.send(msg)
				
			for prim, prim_weight in zip(primitives, data["primitive"]):
				prim.set_weights(prim_weight)
			
			while proc_num == data["node"]:
				#print(data["weights"][0], flush=True)
				expert.set_weights (data["weights"])
				
				id = int(np.random.random()*len(envs))
				cur_env = envs[id]
				cur_prim = primitives[id]
				
				# simulate rollout
				all_s, all_a, all_r, all_mask = distillation.get_rollout(expert, cur_env, cur_prim, rollout_len, expert_action_prob, log_std)
				
				# send rollout back to warehouse
				# and get network weights and update actor
				msg = {"node":proc_num, 
						"s" : all_s,
						"a" : all_a,
						"r" : all_r,
						"mask" : all_mask,
						"request" : ["weights", "node"]}
					
				data = warehouse.send(msg)
				
				
			output_dict['Expert'] = expert
			output_dict['Critic'] = critic
			
		


type_dict = {
		'SimpleActorNode':SimpleActorNode,
		'MixActorNode':MixActorNode,
		'LoadActorNode':LoadActorNode,
		'LoadCriticNode':LoadCriticNode,
		'FreePrimitiveNode':FreePrimitiveNode,
		'SimpleEnvNode':SimpleEnvNode,
		'CartPoleNode':CartPoleNode,
		'DogEnvNode':DogEnvNode,
		'TrainPPONode':TrainPPONode,
		'TrainDistillationNode':new_distillation.TrainDistillationNode,
		'LSTMActorNode':LSTMActorNode
		}

def get_process (type):
	if type not in type_dict:
		raise NameError('Node type {} not known'.format(type))
	
	return type_dict[type]
