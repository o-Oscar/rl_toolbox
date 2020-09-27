import numpy as np
import time
import matplotlib.pyplot as plt
import pybullet as p


class RewardFunc ():
	def set_epoch (self, e):
		pass

class NotDyingsRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.min_base_clearance = 0.08
		
	def step (self):
		base_done = self.state.base_clearance < self.min_base_clearance
		#foot_done = np.any([x < 0.01 for x in self.state.foot_clearance[::2]])
		return -1 if base_done else 0
	
	def done (self):
		base_done = self.state.base_clearance < self.min_base_clearance
		#foot_done = np.any([x < 0.01 for x in self.state.foot_clearance[::2]])
		return base_done
	
	
class SpeedRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		"""
		c0 = 0.5*np.exp(-1)
		c1 = 1*np.exp(-1)
		v0 = np.sqrt(np.sum(np.square(self.state.target_speed)))
		
		self.sigma_2 = 2*v0*c0/c1
		self.a = c0*np.exp(v0*v0/self.sigma_2)
		if 1:
			print("sigma", self.sigma_2)
			print("a", self.a)
		"""
		self.c1 = 1*np.exp(-1)
		self.b = 0.5
	def step (self):
		#return np.exp(-np.sum(np.square(self.state.target_speed - self.state.mean_planar_speed))/self.sigma_2) * self.a
		return -np.sqrt(np.sum(np.square(self.state.target_speed - self.state.mean_planar_speed)))*self.c1 + self.b
	
	def done (self):
		return False
	
class TurningRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		
		self.a = 0.5
		self.sigma_2 = 1
		
	def step (self):
		return np.exp(-np.sum(np.square(self.state.target_rot_speed - self.state.mean_z_rot_speed))/self.sigma_2) * self.a
	
	def done (self):
		return False
	
class PoseRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		#self.target_pose = np.asarray([0.0, 1.0408382989215212, -1.968988857605835]*4)
		self.target_pose = np.asarray([0.0, 0.9, -2*0.9]*4)
		self.pose_min = np.asarray([-0.7318217246298173, -0.8606353612620431, -2.492509298518673]*4)
		self.pose_max = np.asarray([0.7318217246298173, 1.3606960000267483, -0.4816591423398249]*4)
		
		self.a = 1
		
	def step (self):
		#return -np.sum(np.square(self.target_pose - self.state.mean_joint_rot)/np.square(self.pose_max-self.pose_min)) * self.a
		a = self.a #* np.exp(-np.sum(np.square(self.state.mean_planar_speed))/0.5)
		return -np.sum(np.square(self.state.target_pose - self.state.mean_action)) * a
	
	def done (self):
		return False
	
class JointAccRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		
		self.a = 0#1e-6
		
	def step (self):
		return -np.sum(np.square(self.state.acc_joint_rot)) * self.a
	
	def done (self):
		return False
	
class BaseAccRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		
		self.a_0 = 1/200 # / np.sqrt(np.sum(np.square(self.state.target_speed)))
		self.a = self.a_0
	"""
	def set_epoch (self, e):
		#fac = min(max(e/1000, 0), 1)
		fac = 1/(1+np.sum(np.square(self.state.target_speed)))
		self.a = fac * self.a_0
		"""
	def step (self):
		fac = self.a_0 * np.exp(-np.sum(np.square(self.state.mean_planar_speed))/0.5)
		return -np.sum(np.square(self.state.base_pos_acc)) * fac
	
	def done (self):
		return False
	
class FootClearRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		self.a = 0.05
		
	def step (self):
		to_return = 0
		#for i in [1, 3, 5, 7]
		for i in [0, 1, 2, 3]:
			clear = self.state.foot_clearance[i]
			speed = self.state.foot_vel[i]
			if clear < 0.02:
				#to_return -= np.sum(np.square(speed))
				to_return -= np.sum(np.square(speed[:2]))*(1-clear/0.02)
		return to_return*self.a
	
	def done (self):
		return False
	