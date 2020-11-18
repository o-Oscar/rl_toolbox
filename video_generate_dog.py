import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import time

import pybullet as p

from environments.dog_env import DogEnv
from models.actor import SimpleActor, MixtureOfExpert, LSTMActor


if __name__ == '__main__':
	tf.config.threading.set_inter_op_parallelism_threads(1)
	print("inter_op_parallelism_threads : {}".format(tf.config.threading.get_inter_op_parallelism_threads()))
	
	debug = True
	render = False
	load_trained = True
	actor_type = "simple"
	
	env = DogEnv(debug=debug, render=render)


	path = "results\\baseline\\models\\expert\\{}"
	#path = "results\\exp_0\\models\\expert\\{}"
	
	if actor_type=="mix":
		primitives = [SimpleActor(env) for i in range(2)]
		actor = MixtureOfExpert(env, primitives, debug=True)
	elif actor_type == "simple":
		actor = SimpleActor(env)
	elif actor_type == "lstm":
		actor = LSTMActor(env)
	
	if load_trained:
		actor.load(path)
		
	env.test_adr = True
	env.training_change_cmd = False
	env.carthesian_act = True
	#env.state.target_speed =  np.asarray([1, 0])*1
	#env.state.target_rot_speed = 0
	env.training_mode = 1
	obs = env.reset()
	init_state = actor.get_init_state(env.num_envs)
	
	all_rew = []
	all_rew2 = []
	all_done = []
	all_stuff = [[] for i in range(1000)]
	all_obs = []
	all_influence = []
	all_act = []
	all_inf = []
	all_dev = []
	all_speed = []
	all_sim = []
	
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
	env.state.target_speed =  np.asarray([1, 0])*1
	env.state.target_rot_speed = 0
	
	for i in range(1000):
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
		"""
		theta = rot
		speed = np.cos(theta)
		rot = np.sin(theta)
		"""
		#speed = 0
		#rot = 0
		
		"""
		task = [-1, -1,-1, -1, 0, 0, 1, 1, 1, 1, 0, 0]
		rot = task[(i//30)%len(task)]
		"""
		#env.set_cmd(2, rot)
		env.state.target_speed =  np.asarray([speed, 0])
		env.state.target_rot_speed = rot#rot
		
		#print(env.state.base_clearance)
		#print(env.state.target_rot_speed)
		
		obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=1)
		start = time.time()
		#act, init_state = actor.model((obs, init_state))
		act, init_state = actor.model((obs, init_state))
		if actor_type=="mix":
			all_inf.append(actor.inf_model((obs, init_state))[0].numpy())
		dur = time.time()-start
		act = act.numpy()
		all_act.append(act)
		act = act# + np.random.normal(size=12).reshape(act.shape) * np.exp(-3)
		#act = np.asarray([0.0, 1.0408382989215212, -1.968988857605835]*4)
		#act = np.asarray([0.5, 0.5, 0.3]* 4)
		obs, rew, done = env.step(act)
		obs = actor.scaler.scale_obs(obs)
		#obs, rew, done = env.step(np.asarray([0.5, 0.8, 0.3]*4))
		all_obs.append(obs)
		all_rew.append(rew[0])
		all_dev.append(env.dev)
		all_speed.append(env.state.loc_pos_speed[0])
		#print(rew)
		time.sleep(1/30)
		
		sim = np.mean(np.square((obs - env.symetry.state_symetry(obs))))
		all_sim.append(sim)
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
	#print("speed :", np.mean(env.sim.to_plot[8+24]
	print("speed :", np.mean(all_speed))
	print("speed :", np.max(all_speed))
	#print("rew :", np.mean(all_rew2))
	#print("speed :", np.mean(env2.sim.to_plot[8+24]))
	"""
	all_obs = np.asarray(all_obs)[:,0,:]
	plt.plot(all_obs.mean(axis=0))
	plt.plot(all_obs.std(axis=0))
	plt.plot(actor.scaler.mean)
	plt.plot(actor.scaler.std)
	plt.show()
	"""
	"""
	plt.plot(all_rew)
	plt.show()
	
	for i in range(12):
		plt.plot(env.sim.to_plot[i+12])
	#plt.plot(env.sim.to_plot[8+24])
	plt.show()
	
	"""
	for i in range(9):
		plt.plot(env.to_plot[i], label=str(i))
	#plt.plot(env.sim.to_plot[8+24])
	plt.legend()
	plt.show()
	#plt.plot(all_dev)
	#plt.show()
	
	if len(env.sim.raw_frames) > 0:
		with open("results/video/raw.out", "wb") as f:
			f.write(np.stack(env.sim.raw_frames).tostring())
	