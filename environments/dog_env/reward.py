import numpy as np

class FullRewStandard:
	def __init__ (self, state):
		self.all_rew_class = [BodyContactRew, FootLiftingRew, NotInclinedRew, InclinedRew, SpeedRew, RotRew, DoneAt400]#, MaxTorqueRew]#, FootSpeedRew, FootAccRew, TorqueRew] #, TorqueRew, FootSpeedRew, FootAccRew]#, RandDoneRew]
		# self.all_rew_class = [BodyContactRew, NotInclinedRew, SpeedRew, RotRew, InclinedRew, TorqueRew, FootSpeedRew, FootAccRew]
		self.all_rew_inst = [x(state) for x in self.all_rew_class]
		
	def step (self):
		steps = [x.step() * x.a for x in self.all_rew_inst]
		return 1. + np.sum(steps) # 1 
	
	def done (self):
		return bool(np.any([x.done() for x in self.all_rew_inst]))

class BodyContactRew:
	def __init__(self, state):
		self.state = state
		self.a = 1
		
	def step (self):
		return (-1 if self.state.other_contact else 0)
	
	def done (self):
		return self.state.other_contact
	
class FootLiftingRew:
	def __init__(self, state):
		self.state = state
		self.a = 1
		
	def step (self):
		return (-1 if self.done() else 0)
	
	def done (self):
		cur_min_z_raw = self.state.reference_bag.min_z[self.state.phase]
		cur_min_z = (cur_min_z_raw-0.02)*1+0.02
		to_return = np.any([z < mz for (x, y, z), mz in zip(self.state.foot_pos, cur_min_z)])
		# print([mz for (x, y, z), mz in zip(self.state.foot_pos, cur_min_z)])
		# print([z < mz for (x, y, z), mz in zip(self.state.foot_pos, cur_min_z)])
		return to_return
		
		
# TODO : penalize body orientation
class NotInclinedRew:
	def __init__(self, state):
		self.state = state
		self.a = 1
		self.threshold = np.cos(20/180*np.pi)
		
	def step (self):
		return (-1 if self.done() else 0)
	
	def done (self):
		return self.state.loc_up_vect[2] < self.threshold

class InclinedRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = np.cos(20/180*np.pi)
		
	def step (self):
		return (self.state.loc_up_vect[2]-1)/(1-self.threshold)
	
	def done (self):
		return False

class SpeedRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = 0.4 # 0.2
		
	def step (self):
		# return -min(1, np.sum(np.square((self.state.target_speed - self.state.loc_pos_speed[:-1])/self.threshold)))
		return -np.sum(np.square((self.state.target_speed - self.state.loc_pos_speed)/self.threshold))
	
	def done (self):
		return False # np.sum(np.square(self.state.target_speed - self.state.loc_pos_speed[:-1])) > self.threshold**2

class RotRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = 1 # 0.5
		
	def step (self):
		return -np.sum(np.square((self.state.target_rot_speed - self.state.loc_rot_speed)/self.threshold))
	
	def done (self):
		return False # abs(self.state.target_rot_speed - self.state.loc_rot_speed[2]) > self.threshold

class FootPosRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = 0.03
	
	def step (self):
		# dists = np.square(np.asarray(self.state.loc_foot_pos).flatten() - self.state.targ_pos)
		dists = np.square(self.state.mean_loc_foot_pos - self.state.targ_pos)
		# print("->", np.sqrt(np.sum(dists)), np.sum(dists))
		# return -min(1, np.sum(dists)/self.threshold)
		return -np.sum(dists)/self.threshold
	
	def done (self):
		return False

# TODO : penalize acceleration ?

class TorqueRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = 3000 # 1200 # 800
		
	def step (self):
		torque = np.sum(np.abs(self.state.joint_torque))
		# caped = max(torque-150, 0)
		# return -min(1, np.sum(np.abs(self.state.joint_torque))/self.threshold)
		return -min(1, torque/self.threshold)
	
	def done (self):
		return False

class VirtualTorqueRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = 400
		
		dx = 0.2
		dy = 0.2
		dz = 0
		self.targ_pos = np.asarray([dx, dy, dz, dx, -dy, dz, -dx, dy, dz, -dx, -dy, dz]).reshape((4,3))
		
		
	def step (self):
		delta_pos = self.targ_pos - self.state.loc_foot_pos
		# print(np.cross(self.state.foot_force, delta_pos)*1000)
		return -np.sum(np.square(np.cross(self.state.foot_force, delta_pos)))
	
	def done (self):
		return False

class MaxTorqueRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.25
		self.threshold = 11 # 1200 # 800
		
	def step (self):
		return -1 if self.done() else 0
	
	def done (self):
		return np.any(np.abs(self.state.joint_torque) > 20)

class FootSpeedRew:
	def __init__ (self, state):
		self.state = state
		self.a = 0.03
		
		
		r = 0.4
		v = 0.6
		dz = 0.06
		self.m = np.asarray([-v, 0, -dz*2*self.state.f0/r]*4)
		self.M = np.asarray([(1-r)/r*v, 0, dz*2*self.state.f0/r]*4)
		
	def step(self):
		sp = np.asarray(self.state.loc_foot_speed).flatten()
		spm = np.minimum(sp-self.m, 0)
		spM = np.maximum(sp-self.M, 0)
		sp = spm + spM
		return -np.sum(np.square(np.asarray(sp)))
	
	def done (self):
		return False

class FootAccRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.00003
		
	def step (self):
		return -np.sum(np.square(self.state.loc_foot_acc))
	
	def done (self):
		return False

class RandDoneRew:
	def __init__ (self, state):
		self.state = state
		self.mean_steps = 300
		self.a = 1
	def step(self):
		return 0
	def done (self):
		return np.random.random() < 1/self.mean_steps

class DoneAt400:
	def __init__ (self, state):
		self.state = state
		self.a = 1
	def step(self):
		return 0
	def done (self):
		return self.state.frame > 400

"""
class SpeedRew:
	def __init__(self, state):
		self.state = state
		self.a = 1
		self.threshold = 0.4
		
	def step (self):
		return (-1 if self.done() else 0)
	
	def done (self):
		return np.sum(np.square(self.state.target_speed - self.state.loc_pos_speed[:-1])) > self.threshold**2

class RotRew:
	def __init__(self, state):
		self.state = state
		self.a = 1
		self.threshold = 1
		
	def step (self):
		return (-1 if self.done() else 0)
	
	def done (self):
		return abs(self.state.target_rot_speed - self.state.loc_rot_speed[2]) > self.threshold
"""