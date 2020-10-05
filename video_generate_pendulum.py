import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import time

#import pybullet as p

#from environments.dog_env import DogEnv
from environments.simple_env import SimpleEnv
from environments.cartpole import CartPoleEnv

from models.actor import SimpleActor, MixtureOfExpert

#import blindfold

if __name__ == '__main__':
	tf.config.threading.set_inter_op_parallelism_threads(1)
	print("inter_op_parallelism_threads : {}".format(tf.config.threading.get_inter_op_parallelism_threads()))
	
	debug = True
	render = False
	load_trained = True
	actor_type = "simple"
	
	#env = SimpleEnv()
	env = CartPoleEnv()

	#path = "results\\default\\models\\{}_{}"
	#path = "results\\good_full\\models\\expert\\{}"
	path = "results\\exp_0\\models\\swingup\\{}"
	
	if actor_type=="mix":
		primitives = [SimpleActor(env) for i in range(2)]
		actor = MixtureOfExpert(env, primitives, debug=True)
	elif actor_type == "simple":
		actor = SimpleActor(env)
	
	
	if load_trained:
		actor.load(path)
	
	env.test_adr = True
	obs = env.reset()
	init_state = actor.get_init_state(env.num_envs)
	
	all_rew = []
	all_done = []
	all_stuff = [[] for i in range(100)]
	all_obs = []
	all_influence = []
	all_act = []
	all_states = []
	all_e = []
	all_inf = []
	
	for i in range(300):
		"""
		events = p.getKeyboardEvents()
		if 113 in events:
		"""
		env.n = 0
		obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=1)
		start = time.time()
		act, init_state = actor.model((obs, init_state))
		if actor_type=="mix":
			all_inf.append(actor.inf_model((obs, init_state))[0].numpy())
		dur = time.time()-start
		act = act.numpy()# * 0 + 0#[0.1, 0.1]
		#print(act)
		all_act.append(act)
		act = act# + np.random.normal(size=act.flatten().shape[0]).reshape(act.shape) * np.exp(-2)
		#act = act*0 + 0.5 + 0.3/2 * (2*((i//50)%2)-1)
		#act = np.asarray([0.0, 1.0408382989215212, -1.968988857605835]*4)
		#act = np.asarray([0.5, 0.5, 0.3]* 4)
		obs, rew, done = env.step(act)
		all_states.append(env.state)
		all_e.append(env.e)
		all_obs.append(obs)
		all_rew.append(rew[0])
		all_done.append(done)
		#print(rew)
	
	print(env.adr.success)
	
	for i, obs in enumerate(np.asarray(all_obs).reshape((-1, env.obs_dim)).T):
		plt.plot(obs, label=str(i))
	plt.legend()
	plt.show()
	
	if actor_type=="mix":
		plt.plot(np.squeeze(all_inf))
		plt.show()
		
	#print(np.any(all_done))
	for i in range(4):
		plt.plot([s[i] for s in all_states])
	plt.plot(all_e)
	plt.show()
	plt.plot(np.asarray(all_act).reshape((-1, env.act_dim)))
	plt.show()
	