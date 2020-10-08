
import pybullet as p

from environments.dog_env import DogEnv
from models.actor import SimpleActor, MixtureOfExpert

from classroom import PPO

import numpy as np

env = DogEnv(debug=True, render=False)
		
env = DogEnv(debug=True, render=False)

	
def get_weights ():
	path = "results\\retrain\\models\\expert\\{}"
			
	primitives = [SimpleActor(env) for i in range(2)]
	actor = MixtureOfExpert(env, primitives, debug=True)

	actor.load(path)
	return actor.get_weights()

weights = get_weights() 


			
primitives = [SimpleActor(env) for i in range(2)]
actor = MixtureOfExpert(env, primitives, debug=True)


trainer = PPO(env, actor, init_log_std=-3)


actor.set_weights(weights)

all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(100)
x = np.mean(all_r)

print(x)
all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(100)
x = np.mean(all_r)

print(x)
all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(100)
x = np.mean(all_r)

print(x)