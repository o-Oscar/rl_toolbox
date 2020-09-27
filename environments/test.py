from dog_env import DogEnv
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt


T = 1 # s
dt = 1/30
dx = 0.2
dz = 0.4
x0 = 0.5
z0 = 0.3
d_phi = [0, np.pi, np.pi, 0]

def leg_action (phi):
	return [x0+dx*np.sin(phi), 0.5, z0+dz*np.cos(phi)]

def full_action (phi):
	return sum([leg_action(phi+d) for d in d_phi], [])

def motion ():
	return np.asarray([full_action(frame*dt*2*np.pi/T) for frame in range(int(T/dt))])

np.save("dog_env/motion/v1/legs", motion())
	



"""
leg_A = np.asarray([[0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
leg_B = np.asarray([1, 0, 0] * 4)

mes_A = np.diag([1, 1, -1, 1, 1, -1, 1, -1, 1, -1])
mes_B = 0
obs_A = np.zeros((10+12*2, 10+12*2))
obs_A[0:10,0:10] = mes_A
obs_A[10:22,10:22] = leg_A
obs_A[22:,22:] = leg_A

if __name__ == "__main__":
	env = DogEnv(True, False)
	env.reset(1)
	obs1, _, _ = env.step(np.asarray([0.5]*12))
	env.reset(-1)
	obs2, _, _ = env.step(np.asarray([0.5]*12))
	print(np.max(obs2-obs1 @ obs_A))
	
	
"""
	