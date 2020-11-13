import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

import config


def is_observed (is_act, state_Id=-1):
	if is_act:
		return [True for i in range(12)]
							
	if state_Id == config.JOINT_POS:
		return [False for i in range(12)]
	elif state_Id == config.JOINT_POS_RAND:
		return [True for i in range(12)]
				
	elif state_Id == config.JOINT_VEL:
		return [False for i in range(12)]
	elif state_Id == config.JOINT_VEL_RAND:
		return [True for i in range(12)]
		
	elif state_Id == config.LOCAL_UP:
		return [False for i in range(3)]
	elif state_Id == config.LOCAL_UP_RAND:
		return [True for i in range(3)]
		
	elif state_Id == config.ROT_VEL:
		return [False for i in range(3)]
	elif state_Id == config.ROT_VEL_RAND:
		return [True for i in range(3)]
		
	elif state_Id == config.POS_VEL_CMD:
		return [True for i in range(2)]
	elif state_Id == config.ROT_VEL_CMD:
		return [True for i in range(1)]
		
	elif state_Id == config.HEIGHT:
		return [False for i in range(1)] # True
		
	elif state_Id == config.POS_VEL:
		return [False for i in range(3)] # True
	elif state_Id == config.MEAN_POS_VEL:
		return [False for i in range(2)]
	elif state_Id == config.MEAN_ROT_VEL:
		return [False for i in range(1)]
	elif state_Id == config.ACT_OFFSET:
		return [False for i in range(12)]
	
	else:
		print("ERROR : invalid obs config")
		print(1/0)

#all_observed = sum([is_observed (False, id) for id in config.state_vect] + [is_observed (True)], []) * config.obs_transition_len
all_observed = sum([is_observed (False, id) for id in config.state_vect], []) * config.obs_transition_len
inp_len = len(all_observed)
out_len = len([x for x in all_observed if x])

act_A = []
a = 0
for observed in all_observed:
	to_add = np.zeros((out_len,))
	if observed:
		to_add[a] = 1
		a += 1
	act_A.append(to_add)
act_A = np.asarray(act_A)


class Blindfold:
	def action_blindfold (self, input_action):
		return input_action @ act_A