"""
from os.path import dirname, join, abspath
import erquy_py

urdf_name = "idefX"
urdf_path = join(dirname(str(abspath(__file__))), "data", urdf_name, urdf_name + ".urdf")
meshes_path = join(dirname(str(abspath(__file__))), "data", urdf_name)

viz = erquy_py.Visualizer(urdf_path, meshes_path)
world = erquy_py.World()
world.loadUrdf (urdf_path, meshes_path)


print("It works !!!")
"""
"""
import gym

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
action, _states = model.predict(obs, deterministic=True)
obs, reward, done, info = env.step(action)
env.render()
if done:
obs = env.reset()

env.close()
"""

from environments.dog_env import DogEnv
from config import Config
import time
import numpy as np

import torch as th
from torch import nn
import student_models

import matplotlib.pyplot as plt

render = True

config = Config("exp_0", models_names=["student/model"])
env = DogEnv(debug=render, use_realistic_generator=True)
# model = student_models.simple_student_model(env.obs_dim)
model = student_models.conv_student_model(env.obs_dim)
model.load_state_dict(th.load(config.models_best_path["student/model"]))

env_setup={
	# "kp":30,
	"foot_f": [0.3]*4,
}

obs = env.reset(env_setup)
all_obs = [obs]
obs_stack = np.expand_dims(np.stack(all_obs, axis=0), axis=0)
print(obs.shape)
print(obs_stack.shape)
action = np.zeros((12,))

if render:
	input()

all_rew = []

to_plot = [[] for i in range(100)]

for i in range(30000 if render else 300):
	start = time.time()
	with th.no_grad():
		action = model(th.tensor(obs_stack.astype(np.float32))).numpy()[0,-1]
	obs, rew, done, _ = env.step(action)

	all_obs.append(obs)
	obs_stack = np.expand_dims(np.stack(all_obs, axis=0), axis=0)

	all_rew.append([r.step()*r.a for r in env.reward.all_rew_inst])
	# print(env.state.target_speed)
	print(env.state.target_rot_speed, env.state.target_speed)

	if not render:
		# to_plot[0].append(env.state.loc_up_vect[0])
		# to_plot[1].append(env.state.loc_up_vect[1])
		# to_plot[2].append(env.state.loc_up_vect[2])
		# to_plot[0].append(env.state.loc_pos_speed[0])
		# to_plot[1].append(env.state.loc_pos_speed[1])
		# to_plot[2].append(env.state.loc_pos_speed[2])
		# to_plot[3].append(env.state.loc_rot_speed[0])
		# to_plot[4].append(env.state.loc_rot_speed[1])
		# to_plot[5].append(env.state.loc_rot_speed[2])
		print(action.shape)
		for i in range(12):
			to_plot[i].append(action[i])

	if render:
		while (time.time()-start < 0.03):
			pass 

if not render:
	for i in range(12):
		plt.plot(to_plot[i])
	plt.show()

	plt.plot(all_rew)
	plt.legend([type(r).__name__ for r in env.reward.all_rew_inst])
	plt.show()

print("working !!")