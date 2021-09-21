
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import h5py


up_vect = np.asarray([0, 0.2, 0.7])
def get_rotation (up_vect):
	up_vect /= np.sqrt(np.sum(np.square(up_vect)))
	z_vect = np.asarray([0, 0, 1])

	rot_dir = np.cross(up_vect, z_vect)
	rot_dir /= np.sqrt(np.sum(np.square(rot_dir)))
	dot = np.dot(up_vect, z_vect)
	r = R.from_rotvec(rot_dir * np.arccos(dot))
	return r


filename = "2021_08_29_15h49m23s_LogFile.hdf5"#.format(dt_string)
path = os.path.join("src", "logs", filename)
print("reading")
f = h5py.File(path, "r")
print("end_reading")

from environments.dog_env import DogEnv
import time
import numpy as np

import matplotlib.pyplot as plt

if False:
	render = False

	env_setup={
		"kp":60,
		"kd_fac": 0.12,
		"base_state" : np.asarray([0, 0, 0.4, 0, 0, 0, 1]),
		"reset_base" : True,
		# "update_phase": False,
		"phase": 0,
		"foot_f": [0.4]*4,
		# "action" : np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
		"gravity": [0, -0., -9.81],
		"push_f": 0,
	}

	env.reset(env_setup)
	all_joint_pos = []
	for i in range(30*3):
		env.step(np.zeros((12,)))
		all_joint_pos.append(env.state.joint_rot)

	all_joint_pos = np.asarray(all_joint_pos)

	plt.plot(f["joint_rot"][:,:3])
	plt.plot(all_joint_pos[:,:3])
	plt.show()

# for key, value in f.items():
# 	print(key)
# 	plt.plot(value[1:])
# 	plt.show()

# print(np.std(f["up_vect"], axis=0)*180/np.pi)
# print(np.mean(f["up_vect"], axis=0)*180/np.pi)
# plt.plot(f["up_vect"])
# plt.show()


render = True
env = DogEnv(debug=render)

while (True):
	print(f.keys())
	# for leg_qpos, up_vect in zip(f["joint_rot"], f["up_vect"]):
	for leg_qpos, up_vect in zip(f["joint_rot"], f["up_vect"]):
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



# TODO:
# check the kps with log 2021_07_12_20h08m00s_LogFile.hdf5 (put down after middle of log)
# and the kds with log 2021_07_12_19h59m34s_LogFile.hdf5 (tracking of simple traj while staying in the air)
# retrain a teacher with the proper parameters and zero cmd velocity
# train a student network
# add the student network to the dog and test it.