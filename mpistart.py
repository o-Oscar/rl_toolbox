# default file for starting the right work

"""
mpiexec -n 10 python mpistart.py default
"""

"""
ssh -L 16006:127.0.0.1:6006 oscar.boutin@saumon.polytechnique.fr
tensorboard --logdir=results/default/tensorboard --host localhost --port 6006
http://localhost:16006 
"""

from mpi4py import MPI
import time
import numpy as np

import warehouse
from classroom import PPO

from environments.dog_env import DogEnv
from environments.simple_env import SimpleEnv
from environments.cartpole import CartPoleEnv

import config


PPO_RANK = 0
WH_RANK = 1
WORK_START_RANK = 2

def get_assignement (comm):
	num_procs = comm.Get_size()
	
	assignement = {}
	assignement["learner"] = 0
	assignement["wh"] = 1
	assignement["worker"] = list(range(2,num_procs))
	return assignement

if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	my_rank = comm.Get_rank()
	my_name = MPI.Get_processor_name()
	
	assignement = get_assignement (comm)
	
	#env = SimpleEnv()
	#env = CartPoleEnv()
	env = DogEnv()
	USE_ADR = env.adr is not None and config.training["use_adr"]

	print("my_rank:", my_rank, flush=True)
	
	warehouse.start_warehouse(comm, my_rank, assignement["wh"])
	
	desired_rollout_nb = config.training["rollout_nb"]
	all_desired_rollout_nb = []
	min_dumped_rollouts = 100
	last_retrace = -1

	if my_rank == assignement["learner"]:
		trainer = PPO(env, create_save_folder=True)
		
		if config.training["use_init_model"]:
			trainer.restore(config.training["init_model_path"])
		
		start_time = time.time()
		for n in range(config.training["ppo_epoch_nb"]):
			# send the network weights
			# and get the latest rollouts
			msg = {"weights" : trainer.get_weights(), "rollout_nb":desired_rollout_nb, "request" : ["s", "a", "r", "neglog", "mask", "dumped", "adr"]}
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
			trainer.train_networks(n, all_s, all_a, all_r, all_neglog, all_masks)
			
			# choose the right amount of rollout to be send
			"""
			if n-last_retrace > 2:
				min_dumped_rollouts = min(min_dumped_rollouts, dumped_rollout_nb)
				print("min_dumped_rollouts", min_dumped_rollouts, flush=True)
			if (n-last_retrace-2)%5 == 0 and n-last_retrace > 2:
				desired_rollout_nb += min_dumped_rollouts
				print("desired_rollout_nb", desired_rollout_nb, flush=True)
				last_retrace = n
				min_dumped_rollouts = 100
			"""
			
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
			if config.training["adr_save_interval"]:
				env.adr.save()
		
		warehouse.send({}, work_done=True)
		env.adr.save()
		env.close()
		
			
	if my_rank in assignement["worker"]:
		trainer = PPO(env)
		
		msg = {"request" : ["weights"]}
		data = warehouse.send(msg)
		
		print("worker {} received weights.".format(my_rank), flush=True)
		
		while not warehouse.is_work_done:
			
			# simulate rollout
			#time.sleep(0.3+np.random.random()*0.3)
			env_nb = 1 # int(np.random.random()*10)
			rollout_len = 100
			obs_size = 24
			act_size = 12
			all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout()
			
			# send rollout back to warehouse
			# and get network weights and update actor
			msg = {"s" : all_s,
					"a" : all_a,
					"r" : all_r,
					"neglog" : all_neglog,
					"mask" : all_mask,
					"request" : ["weights", "adr"]}
			if USE_ADR:
				msg["adr"] = env.adr.get_msg()
				
			data = warehouse.send(msg)
			
			trainer.set_weights (data["weights"])
			if USE_ADR:
				env.adr.update(data["adr"])
		
		print("worker {} closing".format(my_rank), flush=True)
		env.close()
					
	
	