"""
cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\raisimLib\\raisimUnity\\win32
RaiSimUnity.exe

python environments\\dog_env_rai\\src\\dog_urdf\\create_urdf.py

conda activate psc_sb
cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\idefX\\v3\\rl_toolbox
python generate_traj.py

"""

import time
import numpy as np
from environments.dog_env import DogEnv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import pickle

env = DogEnv (debug=False)
# target_pos = [np.asarray([0, 0, 0]) for i in range(4)]
last_x = None
def to_minimize (x, i, target_pos, base_state, default_action):
	action = default_action
	action[i*3:i*3+3] = x
	env.reset({"base_state": base_state, "action" : action, "kin_use_reference" : False})
	diff = target_pos[i] - env.state.foot_pos[i]
	return np.sum(np.square(diff))

def ik (target_pos, base_state, default_action):
	to_return = []
	for i in range(4):
		args = (i, target_pos, base_state, default_action)
		to_return.append(minimize(to_minimize, np.zeros((3,)), args, bounds=[(-1, 1), (-1, 1), (-1, 1)]).x)
	return np.concatenate(to_return)

def foot_traj (phi, r, dx, dy, dz):
	n = 0
	while phi < 0:
		phi += np.pi*2
		n -= 1
	while phi >= np.pi*2:
		phi -= np.pi*2
		n += 1
		
	if phi < r*2*np.pi:
		lphi = phi/r
		x = (1-np.cos(lphi/2))/2
		y = (1-np.cos(lphi/2))/2
		z = np.sin(lphi/2)
	else:
		x = 1
		y = 1
		z = 0
	x += n
	y += n
		
	return (x-1/4)*dx, (y-1/4)*dy, z*dz

def get_base_state (t, v_targ):
	return [v_targ[0]*t, v_targ[1]*t, 0.35, 0, 0, 0, 1]
	
def get_base_vel (t, v_targ):
	return [v_targ[0], v_targ[1], 0, 0, 0, 0]
	
default_action = np.zeros((12,))

def calc_target_pos (phase, r, dx, dy, dz, Dx, Dy, Dz):
	# front left and back right
	fx1, fy1, fz1 = foot_traj(phase, r, dx, dy, dz)
	
	# front right and back left
	fx2, fy2, fz2 = foot_traj(phase-np.pi, r, dx, dy, dz)
	fx2 += dx/2
	fy2 += dy/2
	
	target_pos = [[fx1+Dx, fy1+Dy, fz1+Dz], [fx2+Dx, fy2-Dy, fz2+Dz], [fx2-Dx, fy2+Dy, fz2+Dz], [fx1-Dx, fy1-Dy, fz1+Dz]]
	# target_pos = [[x-0.065625, y, z] for x, y, z in target_pos] # python -c "print((0.065234375 + .065625)/2)"
	target_pos = [[x-0.05, y, z] for x, y, z in target_pos] # python -c "print((0. + .065625)/2)"
	return target_pos

def get_pos (t, v_targ, f0, r, dx, dy, dz, Dx, Dy, Dz):
	phase = t*f0*2*np.pi
	
	base_state = get_base_state(t, v_targ)
	
	target_pos = calc_target_pos (phase, r, dx, dy, dz, Dx, Dy, Dz)
	
	ik_action = ik(target_pos, base_state, default_action)
	env.reset({"base_state": base_state, "action" : ik_action, "kin_use_reference" : False})
	qpos = env.state.joint_rot
	# print(qpos[-12:])
	return ik_action, qpos

def get_vel (t, v_targ, f0, r, dx, dy, dz, Dx, Dy, Dz):
	e = 1e-6
	tp = t+e
	tm = t-e
	
	_, qposp = get_pos (tp, v_targ, f0, r, dx, dy, dz, Dx, Dy, Dz)
	_, qposm = get_pos (tm, v_targ, f0, r, dx, dy, dz, Dx, Dy, Dz)
	
	return (qposp-qposm)/(2*e)

# v_targ = 0.4 # m.s-1

# --- params for the foot traj that do not change with velocity ---
Dx = 0.2
Dy = 0.16
Dz = 0.02
r = .4 # ratio foot on ground / total traj
dz = 0.15

def calc_full_ik (f0, v_targ, gen_vel):
	# f0 = f0 # Hz = s-1
	f_sim = 30 # Hz = s-1
	N = int(f_sim/f0)

	# --- foot related ---
	dx = v_targ[0] / f0
	dy = v_targ[1] / f0

	to_plot = [[] for i in range(100)]

	all_actions = []

	all_qpos = []
	all_qvel = []
	# all_min_z = []


	for frame in range(N):
		t = frame/N/f0
		
		ik_action, sqpos = get_pos (t, v_targ, f0, r, dx, dy, dz, Dx, Dy, Dz)
		# default_action = ik_action
		all_actions.append(ik_action)
		all_qpos.append(get_base_state(0, v_targ) + list(sqpos))
		
		if gen_vel:
			sqvel = get_vel (t, v_targ, f0, r, dx, dy, dz, Dx, Dy, Dz)
			all_qvel.append(get_base_vel(0, v_targ) + list(sqvel))

	all_qpos = np.stack(all_qpos)
	if gen_vel:
		all_qvel = np.stack(all_qvel)
	else:
		all_qvel = 0
	
	return all_qpos, all_qvel, all_actions

def calc_min_z (f0):
	f_sim = 30 # Hz = s-1
	N = int(f_sim/f0)
	
	min_z = []
	
	for frame in range(N):
		phase = frame*2*np.pi/N
		dx = 0
		dy = 0
		target_pos = calc_target_pos (phase, r, dx, dy, dz, Dx, Dy, Dz)
		target_z = [z for x, y, z in target_pos]
		min_z.append([z - 0.15 for z in target_z])
		
	min_z = np.stack(min_z)
	return min_z

generate_full = True
# py-spy record -o profile.svg --native python3 generate_traj.py


# names = ["forward", "left", "back", "right"]
names = ["in_place"]
all_v_targ = [[0, 0], [0.4, 0.], [0., 0.4], [-0.4, 0.], [0., -0.4]]
f0 = 1. # Hz

references = {}
min_z = calc_min_z(f0)

for name, v_targ in zip(names, all_v_targ):
	qpos, qvel, actions = calc_full_ik(f0, v_targ, generate_full)
	references[name] = (qpos, qvel)

	if not generate_full:
		N = 30
		for frame in range(100000000):
			phase = frame/N*2*np.pi
			t = frame/N/f0
			base_state = get_base_state (t, v_targ)
			ik_action = actions[frame%len(actions)]
			env.reset({"base_state": base_state, "action" : ik_action, "kin_use_reference" : False})
			time.sleep(0.03)


if generate_full:
	to_save = {"f0" : f0, "references" : references, "min_z" : min_z, "kin_use_reference" : False}
	with open(os.path.join("environments", "dog_env", "src", "motion", "walk.txt"), "wb") as f:
		pickle.dump(to_save, f)


	with open(os.path.join("environments", "dog_env", "src", "motion", "walk.txt"), "rb") as f:
		saved = pickle.load(f)
		print(saved)

"""
np.save(os.path.join("environments", "dog_env_rai", "src", "motion", "walk", "qpos.npy"))
np.save(os.path.join("environments", "dog_env_rai", "src", "motion", "walk", "qvel.npy"))
"""


	