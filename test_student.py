
from environments.dog_env import DogEnv
from environments.dog_env.obs_gen import RealisticObsGenerator
from config import Config
import time
import numpy as np
from models import StudentModule

import torch as th
from torch import nn

import matplotlib.pyplot as plt

render = True

config = Config("exp_3", models_names=["student/model"])
env = DogEnv(debug=render)
obs_gen = RealisticObsGenerator(env.state)
# model = student_models.simple_student_model(env.obs_dim)
model = StudentModule(env.get_box_space(obs_gen.obs_dim))
model.load_state_dict(th.load(config.models_best_path["student/model"]))

env_setup={
	"kp":60,
	"kd_fac": 0.12,
	# "base_state" : np.asarray([0, 0, 0.4, 0, 0, 0, 1]),
	# "reset_base" : True,
	# "update_phase": False,
	# "phase": np.pi,
	"foot_f": [0.1]*4,
	# "action" : np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
	"gravity": [0, 0, -9.1],
}

obs = env.reset(env_setup)
print(env.state.foot_f)
obs_gen.reset()
obs = obs_gen.generate()
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
		action = model(th.tensor(obs_stack.astype(np.float32)))[0].numpy()[0,-1]
		# action = action*0
	obs, rew, done, _ = env.step(action)
	obs = obs_gen.generate()

	all_obs.append(obs)
	obs_stack = np.expand_dims(np.stack(all_obs, axis=0), axis=0)

	all_rew.append([r.step()*r.a for r in env.reward.all_rew_inst])
	# print(env.state.target_speed)
	# print(env.state.target_rot_speed, env.state.target_speed)

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