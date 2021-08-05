
from environments.dog_env import DogEnv
from config import Config
import time
import numpy as np

from my_ppo import MyPPO, TeacherActorCriticPolicy
import itertools

import matplotlib.pyplot as plt

config = Config("exp_3", models_names=["teacher/PPO"])
env = DogEnv(debug=False)

model = MyPPO.load(config.models_best_path["teacher/PPO"], env=env, policy=TeacherActorCriticPolicy)


def exec_test (env_setup):
	obs = env.reset(env_setup)
	action = np.zeros((12,))

	all_rew = []
	for i in range(400):
		start = time.time()
		action, _states = model.predict(obs, deterministic=True)

		# sym_obs = obs @ env.obs_gen.get_sym_obs_matrix()
		# sym_action, _states = model.predict(sym_obs, deterministic=True)
		# action = switch_legs @ sym_action
		
		action = action + np.random.normal(size=12) * np.exp(-3)
		obs, rew, done, _ = env.step(action)
		all_rew.append([r.step()*r.a for r in env.reward.all_rew_inst])
		# print(env.state.target_speed)
		if done:
			return i, (env.state.foot_f, env.state.kp0, env.state.kd0_fac, env.state.gravity)
	return i, (env.state.foot_f, env.state.kp0, env.state.kd0_fac, env.state.gravity)

env_setup={
	"kp":60,
	"kd_fac":0.05,
	# "base_state" : np.asarray([0, 0, 0.6, 0, 0, 0, 1]),
	# "reset_base" : True,
	# "update_phase": False,
	# "phase": np.pi,
	"foot_f": [0.1]*4,
	# "action" : np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
	"default_gravity": True,
}

if True:
	for i in range(100):
		env_setup = {
			"foot_f": [0.1]*4,
			"gravity": [0, 0.1*i, -9.81],
		}
		l, args = exec_test(env_setup)
		if l < 399:
			print(args)
		else:
			print("_")

if False:

	N = 5
	res = np.zeros((N,N))
	for i, j in itertools.product(range(N), range(N)):
		env_setup={
			"kp":60,
			"kd_fac":0.05,
			# "base_state" : np.asarray([0, 0, 0.6, 0, 0, 0, 1]),
			# "reset_base" : True,
			# "update_phase": False,
			# "phase": np.pi,
			"foot_f": [0.3*i/N]*4,
			# "action" : np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
			"gravity": [0, 3*j/N, -9.81],
		}
		print(env_setup)
		res[i,j] = exec_test(env_setup)
	print(res)