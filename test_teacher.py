
from environments.dog_env import DogEnv, DogEnv_follow
from config import Config
import time
import numpy as np

from my_ppo import MyPPO, TeacherActorCriticPolicy

import matplotlib.pyplot as plt


render = False

# config = Config("follow_0", models_names=["follower/PPO"])
# env = DogEnv_follow(debug=False)
# motor_model = MyPPO.load(config.models_best_path["follower/PPO"], env=env, policy=MotorActorCriticPolicy)

config = Config("exp_0", models_names=["teacher/PPO"])
env = DogEnv(debug=render)#, motor_model=motor_model)


env_setup={
	"kp":60,
	"kd_fac":0.12,
	# "base_state" : np.asarray([0, 0, 0.6, 0, 0, 0, 1]),
	# "reset_base" : True,
	"update_phase": False,
	"phase": 0,
	"foot_f": [0.2]*4,
	# "action" : np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
	"gravity": [0, 0, -9.81],
	"qpos": np.array([0, 0, 3.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
}

env.reset(env_setup)

model = MyPPO.load(config.models_best_path["teacher/PPO"], env=env, policy=TeacherActorCriticPolicy)

obs = env.reset()
action = np.zeros((12,))

if render:
	input()

all_rew = []

to_plot = [[] for i in range(100)]
from my_ppo import switch_legs

for i in range(300000 if render else 300):
	start = time.time()
	action, _states = model.predict(obs, deterministic=True)

	# sym_obs = {key:x@y for (key, x), (_, y) in zip(obs.items(), env.obs_gen.get_sym_obs_matrix().items())}

	# sym_action, _states = model.predict(sym_obs, deterministic=True)
	# action = switch_legs @ sym_action
	
	action = action# + np.random.normal(size=12) * np.exp(-3)
	obs, rew, done, _ = env.step(action)
	all_rew.append([r.step()*r.a for r in env.reward.all_rew_inst])
	# print(env.state.target_speed)
	if done:
		print(i)

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
		plt.plot(to_plot[i])
		# plt.plot(to_plot[i+12])
	plt.show()

	plt.plot(all_rew)
	plt.legend([type(r).__name__ for r in env.reward.all_rew_inst])
	plt.show()

print("working !!")