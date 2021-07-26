
from environments.dog_env import DogEnv
from config import Config
import time
import numpy as np

from stable_baselines3 import PPO

import matplotlib.pyplot as plt

render = True

config = Config("friction_0", models_names=["teacher/PPO"])
env = DogEnv(debug=render)



model = PPO.load(config.models_best_path["teacher/PPO"], env=env)

env_setup={
	# "kp":30,
	"base_state" : np.asarray([0, 0, 0.6, 0, 0, 0, 1]),
	"update_phase": False,
	"phase": np.pi,
	"foot_f": [0.5]*4,
}

obs = env.reset(env_setup)
action = np.zeros((12,))

if render:
	input()

all_rew = []

to_plot = [[] for i in range(100)]
from my_ppo import switch_legs

for i in range(30000 if render else 300):
	start = time.time()
	action, _states = model.predict(obs, deterministic=True)

	# sym_obs = obs @ env.obs_gen.get_sym_obs_matrix()
	# sym_action, _states = model.predict(sym_obs, deterministic=True)
	# action = switch_legs @ sym_action
	
	action = action * 0
	obs, rew, done, _ = env.step(action)
	all_rew.append([r.step()*r.a for r in env.reward.all_rew_inst])
	# print(env.state.target_speed)

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
		for i in range(12):
			to_plot[i].append(action[i])
			to_plot[i+12].append(env.state.joint_torque[i])

	if render:
		while (time.time()-start < 0.03):
			pass 

if not render:
	for i in range(12):
		plt.plot(to_plot[i+12])
	plt.show()

	plt.plot(all_rew)
	plt.legend([type(r).__name__ for r in env.reward.all_rew_inst])
	plt.show()

print("working !!")