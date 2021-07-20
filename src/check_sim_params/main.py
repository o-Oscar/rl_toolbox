"""
python3 src/check_sim_params/main.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pickle
from environments.dog_env import DogEnv


if  len(sys.argv) > 1 and sys.argv[1] == "load_logs":
	import h5py
	print()
	for file_name, save_name in [
					("2021_07_12_20h13m51s_LogFile.hdf5", "log_kp"), 
					("2021_07_12_19h59m34s_LogFile.hdf5", "log_kd"),
				]:
		file_path = os.path.join("src", "logs", file_name)
		f = h5py.File(file_path, "r")
		to_save = {}
		for key, value in f.items():
			to_save[key] = np.asarray(value)
		with open(os.path.join("src", "logs", save_name), "wb") as save_file:
			save_file.write(pickle.dumps(to_save))

	exit()

if  len(sys.argv) > 1 and sys.argv[1] == "check_kp":
	with open(os.path.join("src", "logs", "log_kp"), "rb") as f:
		log = pickle.load(f)
	# print(log.keys())
	# plt.show()

	env = DogEnv()
	low_kp = 30
	high_kp = 80
	for i in range(10):
		mid_kp = (low_kp+high_kp)/2
		env_setup = {
			"kp": mid_kp,
			"update_phase": False,
		}
		env.reset(env_setup)

		all_leg_pos = []
		all_targets = []
		for t in range(60):
			action = np.zeros((12,))
			env.step(action)
			all_leg_pos.append(env.state.joint_rot)
			all_targets.append(env.state.joint_target)
		
		all_leg_pos = np.asarray(all_leg_pos)
		all_targets = np.asarray(all_targets)
		

		if all_leg_pos[-1, 2] > log["qpos"][-1,2]:
			high_kp = mid_kp
		else:
			low_kp = mid_kp

	print("Best Kp : ", (low_kp+high_kp)/2)
	plt.plot(log["qpos"][80:,])
	plt.plot(all_leg_pos)
	# plt.plot(all_targets)
	plt.show()
	
	exit()

if  len(sys.argv) > 1 and sys.argv[1] == "check_kd":
	with open(os.path.join("src", "logs", "log_kd"), "rb") as f:
		log = pickle.load(f)
	print(log.keys())

	env = DogEnv()
	for i in range(1):
		env_setup = {
			"kd_fac": 0.12,
			"update_phase": True,
			"reset_base": True,
			"base_state": np.asarray([0, 0, 1] + [0, 0, 0, 1])
		}
		env.reset(env_setup)

		all_leg_pos = []
		all_targets = []
		for t in range(90):
			action = np.zeros((12,))
			env.step(action)
			all_leg_pos.append(env.state.joint_rot)
			all_targets.append(env.state.joint_target)
		
		all_leg_pos = np.asarray(all_leg_pos)
		all_targets = np.asarray(all_targets)
		

	joint = 1
	plt.plot(log["qpos"][:,joint])
	plt.plot(all_leg_pos[:,joint])
	plt.plot(all_targets[:,joint])
	plt.show()


	exit()

print()
print("Usage : python3 src/check_sim_params/main.py <program>")
print()
print("\t check_kp : Reads a log where the dog is put standing on the ground and compares it to simulation to get the best kp")
print("\t check_kd : Reads a log where the dog is doing simple movement in the air and compares it to simulation to get the best kd")
print("\t load_logs : Reads the necessary logs into pickle to avoid importing h5py (slow as hell for no reason)")
exit()





filename = "2021_07_12_19h59m34s_LogFile.hdf5"#.format(dt_string)
path = os.path.join("src", "logs", filename)
print("reading")
f = h5py.File(path, "r")
print("end_reading")

from environments.dog_env import DogEnv
import time
import numpy as np

import matplotlib.pyplot as plt

render = True

env = DogEnv(debug=render)
input()
"""
for key, value in f.items():
	print(key)
	plt.plot(value[1:])
	plt.show()
"""
while (True):
	for leg_qpos, up_vect in zip(f["qpos"], f["up_vect"]):
		# "base_state" in self.state.sim_args and "leg_pos"
		h0 = 1
		body_r = get_rotation(np.asarray([up_vect]))
		base_state = np.asarray([0, 0, h0] + list(body_r.as_quat().flatten()))
		env.reset({"leg_pos":leg_qpos, "base_state":base_state}, render=False)

		min_foot_h = h0
		for i in range(4):
			min_foot_h = min(min_foot_h, env.state.foot_pos[i][2])
		h = h0 - min_foot_h + 0.02
		base_state = np.asarray([0, 0, h] + list(body_r.as_quat().flatten()))
		env.reset({"leg_pos":leg_qpos, "base_state":base_state})

		time.sleep(0.03)
	print("Looping")


