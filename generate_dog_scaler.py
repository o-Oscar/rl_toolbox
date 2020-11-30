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
	debug = True
	render = False
	load_trained = True
	actor_type = "simple"
	
	env = DogEnv(debug=debug, render=render)


	#path = "results\\baseline\\models\\expert\\{}"
	path = "results\\exp_0\\models\\expert\\{}"
	
	if actor_type=="mix":
		primitives = [SimpleActor(env) for i in range(2)]
		actor = MixtureOfExpert(env, primitives, debug=True)
	elif actor_type == "simple":
		actor = SimpleActor(env)
	elif actor_type == "lstm":
		actor = LSTMActor(env)
	
	if load_trained:
		actor.load(path)
	
	print(env.blindfold.act_A.shape)
	print(actor.scaler.mean.shape)
	print((actor.scaler.mean @ env.blindfold.act_A).shape)
	
	mean_blind = actor.scaler.mean @ env.blindfold.act_A
	std_blind = actor.scaler.std @ env.blindfold.act_A
	np.save(path.format("scaler_mean_blind") + ".npy", mean_blind)
	np.save(path.format("scaler_std_blind") + ".npy", std_blind)
	
	
	
	