import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

import config
	
def get_A (is_act, state_Id=-1):
	if is_act:
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
							
	if state_Id == config.JOINT_POS or state_Id == config.JOINT_POS_RAND:
		return np.asarray([[0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	elif state_Id == config.LOCAL_UP or state_Id == config.LOCAL_UP_RAND:
		return np.diag([1, -1, 1])
	elif state_Id == config.ROT_VEL or state_Id == config.ROT_VEL_RAND:
		return np.diag([-1, 1, -1])
	elif state_Id == config.POS_VEL_CMD:
		return np.diag([1, -1])
	elif state_Id == config.ROT_VEL_CMD:
		return np.diag([-1])
	elif state_Id == config.HEIGHT:
		return np.diag([1])
	elif state_Id == config.POS_VEL:
		return np.diag([1, -1, 1])
	elif state_Id == config.MEAN_POS_VEL:
		return np.diag([1, -1])
	elif state_Id == config.MEAN_ROT_VEL:
		return np.diag([-1])
	else:
		print("ERROR : invalid obs config")
		print(1/0)

def get_B (is_act, state_Id=-1):
	if is_act:
		return np.asarray([0, 1, 0] * 4)
							
	if state_Id == config.JOINT_POS or state_Id == config.JOINT_POS_RAND:
		return np.zeros((12,))
	elif state_Id == config.LOCAL_UP or state_Id == config.LOCAL_UP_RAND:
		return np.zeros((3,))
	elif state_Id == config.ROT_VEL or state_Id == config.ROT_VEL_RAND:
		return np.zeros((3,))
	elif state_Id == config.POS_VEL_CMD:
		return np.zeros((2,))
	elif state_Id == config.ROT_VEL_CMD:
		return np.zeros((1,))
	elif state_Id == config.HEIGHT:
		return np.zeros((1,))
	elif state_Id == config.POS_VEL:
		return np.zeros((3,))
	elif state_Id == config.MEAN_POS_VEL:
		return np.zeros((2,))
	elif state_Id == config.MEAN_ROT_VEL:
		return np.zeros((1,))
	else:
		print("ERROR : invalid obs config")
		print(1/0)

act_A = get_A (True)
act_B = get_B (True)


all_trans_A = [get_A (False, id) for id in config.state_vect] + [get_A (True)]
all_trans_B = [get_B (False, id) for id in config.state_vect] + [get_B (True)]

obs_len = np.sum([A.shape[0] for A in all_trans_A]) * config.obs_transition_len
obs_A = np.zeros((obs_len, obs_len))
obs_B = np.zeros((obs_len, ))
a = 0
for i in range(config.obs_transition_len):
	for A, B in zip(all_trans_A, all_trans_B):
		b = a + A.shape[0]
		obs_A[a:b,a:b] = A
		obs_B[a:b] = B
		a = b
		

class Symetry:
	def action_symetry (self, input_action):
		return input_action @ act_A + act_B

	def state_symetry (self, input_obs):
		return input_obs @ obs_A  + obs_B

	def loss (self, actor, input_obs, init_state, mask):
		diff = actor.model((input_obs, init_state))[0] - self.action_symetry(actor.model((self.state_symetry(input_obs), init_state))[0])
		return tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(diff), axis=-1), mask))/tf.reduce_mean(mask)

