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

from models.actor import Actor

#import blindfold

if __name__ == '__main__':
	tf.config.threading.set_inter_op_parallelism_threads(1)
	print("inter_op_parallelism_threads : {}".format(tf.config.threading.get_inter_op_parallelism_threads()))
	
	debug = True
	render = False
	load_trained = True
	
	#env = SimpleEnv()
	env = CartPoleEnv()

	#path = "results\\default\\models\\{}_{}"
	path = "results\\exp_0\\models\\{}_{}"
	
	actor = Actor(env)
	if load_trained:
		actor.load_primitive(path)
		actor.load_influence(path)
	
	obs = env.reset(zero=False, c_mode=0)
	init_state = actor.get_init_state(env.num_envs)
	
	all_rew = []
	all_done = []
	all_stuff = [[] for i in range(100)]
	all_obs = []
	all_influence = []
	all_act = []
	all_states = []
	all_e = []
	
	for i in range(300):
		"""
		events = p.getKeyboardEvents()
		speed = 1
		rot = 0
		if 113 in events:
			rot += 1
		if 100 in events:
			rot -= 1
		if 115 in events:
			speed -=1
		if 122 in events:
			speed +=1
		"""
		"""
		if speed == 0:
			rot = 0
		"""
		"""
		speed = 0
		rot = 0
		"""
		env.n = 0
		obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=1)
		start = time.time()
		act, init_state = actor.model((obs, init_state))
		dur = time.time()-start
		act = act.numpy()# * 0 + 0#[0.1, 0.1]
		#print(act)
		all_act.append(act)
		act = act# + np.random.normal(size=act.flatten().shape[0]).reshape(act.shape) * np.exp(-3)
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
	
	
	for i, obs in enumerate(np.asarray(all_obs).reshape((-1, env.obs_dim)).T):
		plt.plot(obs, label=str(i))
	plt.legend()
	plt.show()
	
	print(np.any(all_done))
	for i in range(4):
		plt.plot([s[i] for s in all_states])
	plt.plot(all_e)
	plt.show()
	plt.plot(np.asarray(all_act).reshape((-1, env.act_dim)))
	plt.show()
	