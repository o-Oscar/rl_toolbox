import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import time

import pybullet as p

from environments.dog_env import DogEnv
from models.actor import SimpleActor, MixtureOfExpert


if __name__ == '__main__':
	tf.config.threading.set_inter_op_parallelism_threads(1)
	print("inter_op_parallelism_threads : {}".format(tf.config.threading.get_inter_op_parallelism_threads()))
	
	debug = True
	render = False
	load_trained = True
	actor_type = "simple"
	
	env = DogEnv(debug=debug, render=render)


	path = "results\\exp_1\\models\\expert\\{}"
	
	if actor_type=="mix":
		primitives = [SimpleActor(env) for i in range(2)]
		actor = MixtureOfExpert(env, primitives, debug=True)
	elif actor_type == "simple":
		actor = SimpleActor(env)
	
	if load_trained:
		actor.load(path)
		
	env.test_adr = False
	obs = env.reset()
	init_state = actor.get_init_state(env.num_envs)
	
	all_rew = []
	all_rew2 = []
	all_done = []
	all_stuff = [[] for i in range(100)]
	all_obs = []
	all_influence = []
	all_act = []
	all_inf = []
	
	"""
	env.reset(1)
	obs1, _, _ = env.step(np.asarray([0.5]*12))
	obs1 = np.expand_dims(np.asarray(obs1, dtype=np.float32), axis=1)
	act1, init_state = actor.model((obs1, init_state))
	print(act1)
	env.reset(-1)
	obs2, _, _ = env.step(np.asarray([0.5]*12))
	obs2 = np.expand_dims(np.asarray(obs2, dtype=np.float32), axis=1)
	act2, init_state = actor.model((obs2, init_state))
	print(act2)
	
	
	
	print(1/0)
	"""
	for i in range(30*15):
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
		if speed == 0:
			rot = 0
		"""
		
		#speed = 1
		#rot = -1
		
		"""
		task = [-1, -1,-1, -1, 0, 0, 1, 1, 1, 1, 0, 0]
		rot = task[(i//30)%len(task)]
		"""
		#env.set_cmd(2, rot)
		
		env.state.target_speed =  np.asarray([1, 0])*speed
		env.state.target_rot_speed = rot
		
		
		
		
		obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=1)
		start = time.time()
		#act, init_state = actor.model((obs, init_state))
		act, init_state = actor.model((obs, init_state))
		if actor_type=="mix":
			all_inf.append(actor.inf_model((obs, init_state))[0].numpy())
		dur = time.time()-start
		act = act.numpy()
		all_act.append(act)
		act = act #+ np.random.normal(size=12).reshape(act.shape) * np.exp(-3)
		#act = np.asarray([0.0, 1.0408382989215212, -1.968988857605835]*4)
		#act = np.asarray([0.5, 0.5, 0.3]* 4)
		obs, rew, done = env.step(act)
		all_obs.append(obs)
		all_rew.append(rew[0])
		#print(rew)
		time.sleep(1/30)
	"""
	all_act = np.asarray(all_act).reshape((-1, 12))
	for i in range(4):
		plt.plot(all_act[:,0+3*i], all_act[:,2+3*i], 'o')
	#plt.plot(all_act[:,1])
	
	for i in range(4):
		l = [np.sum(np.square(v)[:2]) if d < 0.02 else 0 for d, v in zip(env.sim.to_plot[i+24], env.sim.to_plot[i+24+8+2])]
		#plt.plot(l, label=str(i))
		plt.plot(env.sim.to_plot[i+24], label=str(i)+"dist")
		#plt.plot(np.sqrt(np.sum(np.square(env.sim.to_plot[i+24+8+2])[:,:2], axis=1)), label=str(i)+"speed")
		
	plt.legend()
	
	if actor_type=="mix":
		plt.plot(np.squeeze(all_inf))
		plt.show()
		"""
	
	print("rew :", np.mean(all_rew))
	#print("speed :", np.mean(env.sim.to_plot[8+24]))
	print("rew :", np.mean(all_rew2))
	#print("speed :", np.mean(env2.sim.to_plot[8+24]))
	"""
	for i in range(7):
		plt.plot(env.to_plot[i])
	plt.plot(env.sim.to_plot[8+24])
	
	
	plt.show()
	"""
	if len(env.sim.raw_frames) > 0:
		with open("results/video/raw.out", "wb") as f:
			f.write(np.stack(env.sim.raw_frames).tostring())
	