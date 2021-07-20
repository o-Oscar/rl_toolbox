import numpy as np

class FullRewStandard:
	def __init__ (self, state):
		# self.all_rew_class = [NotDyingsRew, PoseRew, Rspeed, NormalForceRew, RewRotSpeed, Rb]
		# self.all_rew_class = [NotDyingsRew, PoseRew, Rspeed, NormalForceRew, RewRotSpeed, Rb, TargetSmoothnessRew, TorqueRew]
		# self.all_rew_class = [BodyContactRew, PoseRew, Rspeed, NormalForceRew, RewRotSpeed, Rb, TargetSmoothnessRew, TorqueRew]
		# self.all_rew_class = [BodyContactRew, PoseRew, Rspeed, NormalForceRew, RewRotSpeed, Rb, TargetSmoothnessRew, TorqueRew]
		# self.all_rew_class = [StayOnGround, BodyContactRew, PoseRew, Rspeed, RrotSpeed, Rb, NormalForceRew, TargetSmoothnessRew, TorqueRew, FootClearanceRew]
		self.all_rew_class = [BodyContactRew]
		# self.all_rew_class = [BodyContactRew, PoseRew, Rspeed, NormalForceRew, RewRotSpeed, Rb, TorqueRew]
		self.all_rew_inst = [x(state) for x in self.all_rew_class]
		
	def step (self):
		return np.sum([x.step() * x.a for x in self.all_rew_inst])
	
	def done (self):
		return bool(np.any([x.done() for x in self.all_rew_inst]))

class BodyContactRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.5
		
	def step (self):
		return (0 if self.state.other_contact else 1) - 0.44*2 # - 0.52*2
	
	def done (self):
		return self.state.other_contact
	
class NotDyingsRew:
	def __init__(self, state):
		self.state = state
		self.a = 0.5
		
	def step (self):
		return 0
	
	def done (self):
		return self.state.loc_up_vect[2] < 0.5

class StayOnGround:
	def __init__(self, state):
		self.state = state
		self.a = 0.
	def step (self):
		return 0
	def done (self):
		return not (self.state.ground_limits[0] < self.state.base_pos[0] < self.state.ground_limits[1] and self.state.ground_limits[2] < self.state.base_pos[1] < self.state.ground_limits[3])
	

class PoseRew:
	def __init__(self, state):
		self.state = state
		
		self.mask = np.asarray([1, 1, 1] * 4)
		# self.a = 0.05 # 0.02
		self.a = 0.1 / 3# 0.02
		
	def step (self):
		a = 1 # np.exp(-np.sum(np.square(self.state.mean_planar_speed))/0.5)
		return -np.sum(np.square(self.state.target_pose - self.state.mean_action) * self.mask) * a
	
	def done (self):
		return False

class Rspeed:
	def __init__ (self, state):
		self.state = state
		self.max_vel = 0.4
		self.a = 0.05 # * 2
	
	def step (self):
		
		norm = np.sqrt(np.sum(np.square(self.state.target_speed)))
		if norm < 0.01:
			return 1
		
		dir = self.state.target_speed / norm
		
		loc_v = np.asarray(self.state.loc_pos_speed[:2])
		vpr = np.sum(loc_v * dir)
		if vpr > self.max_vel:
			return 1
		else:
			# return np.exp(-(1/0.02) * np.square(vpr - self.max_vel))
			return np.exp(-2 * np.square(vpr - self.max_vel))
		
	def done (self):
		return False
	

class RrotSpeed:
	def __init__ (self, state):
		self.state = state
		self.max_vel = 0.6
		self.a = 0.05
		
	def step (self):
		return np.exp(-1.5 * np.square(self.state.base_rot_speed[2] - self.state.target_rot_speed))
	
	def done (self):
		return False


class Rb:
	def __init__ (self, state):
		self.state = state
		self.max_vel = 0.6
		self.a = 0.04
		self.c = 0
		self.to_plot = [[] for i in range(100)]
		
	def step (self):
		norm_vel = np.sqrt(np.sum(np.square(self.state.target_speed)))
		norm_rot = self.state.target_rot_speed
		if norm_vel < 0.01:
			v0_2 = np.square(norm_vel)
		else:
			dir_vel = self.state.target_speed / norm_vel
			loc_v = np.asarray(self.state.loc_pos_speed[:2])
			proj_dir = dir_vel * np.sum(loc_v * dir_vel)
			v0_2 = np.sum(np.square(loc_v - proj_dir)) + np.square(self.state.base_pos_speed[2])
		speed_rew = np.exp(-1.5 * v0_2)
		rot_rew = np.exp(-1.5 * np.sum(np.square(self.state.base_rot_speed[:2])))
		"""
		if self.c%2 == 0:
			self.to_plot[0].append(self.state.base_rot_speed[0])
			self.to_plot[1].append(self.state.base_rot_speed[1])
		self.c += 1
		"""
		return speed_rew + rot_rew
	
	def done (self):
		return False


class NormalForceRew:
	def __init__ (self, state):
		self.state = state
		self.a = 0.05 # 0.2
		self.dphi = [0, np.pi, np.pi, 0]
		# self.rs = [0, 0, 0, 0]
		self.rs = [-0.5, -0.5, -0.5, -0.5]
	
	def step (self):
		rew = 0
		for i in range(4):
			c = np.cos(self.state.foot_phases[0]-self.dphi[i])
			if c < self.rs[i]:
				rew -= self.state.foot_force[i]/0.2
			else:
				pass
				# rew += self.state.foot_force[i]/0.2
		return rew
	def done (self):
		return False

"""
# trot
dphi = [0, np.pi, np.pi, 0]
rs = [0, 0, 0, 0]
"""
# walk
dphi = [np.pi/2, np.pi*3/2, 0, np.pi]
s = np.sqrt(2)/2
rs = [-s, -s, -s, -s]


class FootLiftingRew:
	def __init__ (self, state):
		self.state = state
		self.a = 0.01 # 0.2
		self.dphi = dphi
		self.rs = rs
		#self.rs = [-0.5, -0.5, -0.5, -0.5]
	
	def step (self):
		rew = 0
		for i in range(4):
			c = np.cos(self.state.foot_phases[0]-self.dphi[i])
			if c < self.rs[i]:
				rew -= 0.5 if self.state.foot_force[i] > 0.001 else 0
			else:
				pass
				# rew += self.state.foot_force[i]/0.2
		return rew
	def done (self):
		return False

class FootClearanceRew:
	def __init__ (self, state):
		self.state = state
		self.a = 0.01 # 0.2
		self.dphi = dphi
		self.rs = rs
		self.max_clear = 0.03
		
		#self.rs = [-0.5, -0.5, -0.5, -0.5]
	
	def step (self):
		rew = 0
		for i in range(4):
			c = np.cos(self.state.foot_phases[0]-self.dphi[i])
			if c < self.rs[i]:
				rew += min((np.min(self.state.foot_scans[9*i:9*(i+1)])-0.02)/self.max_clear, 1)
			else:
				pass
				rew -= (np.min(self.state.foot_scans[9*i:9*(i+1)])-0.02)/self.max_clear
		return rew
	def done (self):
		return False

class TargetSmoothnessRew:
	def __init__ (self, state):
		self.state = state
		self.a = 0.04 * 2 * 3 # 0.025 # * 0.3
		#self.all_rew = []
	def step (self):
		if False: #True:
			to_return = 0
			for i in range(4):
				to_return += np.linalg.norm(self.state.prev_foot_target[0][i*3:i*3+3] - 2*self.state.prev_foot_target[1][i*3:i*3+3] + self.state.prev_foot_target[2][i*3:i*3+3])
		else:
			to_return = np.linalg.norm(self.state.prev_foot_target[0] - 2*self.state.prev_foot_target[1] + self.state.prev_foot_target[2])
			#self.all_rew.append(to_return)
		return -to_return
	def done (self):
		return False



class TorqueRew:
	def __init__ (self, state):
		self.state = state
		self.a = 2e-5 # * 0.3
	def step (self):
		return -np.sum(np.abs(self.state.joint_torque))
	def done (self):
		return False
