
import os

from models.actor import SimpleActor

class SimpleActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		actor = SimpleActor (env)
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
			
class FreePrimitiveNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		actor = input_dict['Actor'][0]
		for prim in actor.primitives_cpy:
			for layer in prim.layers:
				layer.trainable = True if self.data['free_prop'] == "1" else False
		output_dict['Actor'] = actor

		if self.mpi_role == 'main':
			
			#save the actor
			actor.save(actor.save_path)
			
from environments import cartpole

class CartPoleNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		env = cartpole.CartPoleEnv()
		env.mode = int(self.data['mode_prop'])
		output_dict['Env'] = env

from classroom import PPO
import warehouse
import time
import numpy as np

class TrainPPONode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		USE_ADR = True
		# to test (as a worker) if we keep going : data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
		
		env = input_dict['Environment'][0]
		actor = input_dict['Actor'][0]
		
		if self.mpi_role == 'main':
			tensorboard_path = os.path.join(save_path['tensorboard'], self.data['tensorboard_name_prop'])
			os.makedirs(tensorboard_path)
			
			trainer = PPO(env, actor, tensorboard_path)
			trainer.model_save_interval = int(self.data['model_save_interval_prop'])
			train_step_nb = int(self.data['train_step_nb_prop'])
			if "Critic" in input_dict:
				trainer.critic = input_dict['Critic'][0]
			
			start_time = time.time()
			desired_rollout_nb = int(self.data['rollout_nb_prop'])
			for n in range(int(self.data['epoch_nb_prop'])):
				# send the network weights
				# and get the latest rollouts
				msg = {"node":proc_num, "weights" : trainer.get_weights(), "rollout_nb":desired_rollout_nb, "request" : ["s", "a", "r", "neglog", "mask", "dumped", "adr"]}
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
				trainer.train_networks(n, all_s, all_a, all_r, all_neglog, all_masks, train_step_nb)
				
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
			trainer = PPO(env, actor)
			rollout_len = int(self.data['rollout_len_prop'])
			#data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
			msg = {"request" : ["weights", "node"]}
			data = warehouse.send(msg)
			
			#print("worker {} received weights.".format(my_rank), flush=True)
			
			while proc_num > data["node"]:
				time.sleep(0.3)
				print("sleeping", flush=True)
			
			while proc_num == data["node"]:
				
				trainer.set_weights (data["weights"])
				
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
				if USE_ADR:
					msg["adr"] = env.adr.get_msg()
					
				data = warehouse.send(msg)
				
				if USE_ADR:
					env.adr.update(data["adr"])
				
			
			#print("worker {} closing".format(my_rank), flush=True)
		
			#trainer.set_weights (data["weights"])
			#output_dict['Trained actor'] = input_dict['Actor'][0]
		
		
		output_dict['Trained actor'] = trainer.actor
		output_dict['Critic'] = trainer.critic



type_dict = {
		'SimpleActorNode':SimpleActorNode,
		'MixActorNode':MixActorNode,
		'LoadActorNode':LoadActorNode,
		'FreePrimitiveNode':FreePrimitiveNode,
		'CartPoleNode':CartPoleNode,
		'TrainPPONode':TrainPPONode,
		}

def get_process (type):
	if type not in type_dict:
		raise NameError('Node type {} not known'.format(type))
	
	return type_dict[type]
