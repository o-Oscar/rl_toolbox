import numpy as np
from scipy.spatial.transform import Rotation as R

class ObsGen:
	def reset (self):
		pass

class TeacherObsGenerator (ObsGen):
	def __init__ (self, state):
		self.state = state

		
		self.sub_gen_class = {"real_obs": RealisticObsGenerator, "sim_obs": SimObsGenerator, "vf_obs": VfGenerator}
		self.sub_gen = {key:Gen(self.state) for key, Gen in self.sub_gen_class.items()}
		
		self.obs_dim = {key:gen.obs_dim for key, gen in self.sub_gen.items()}
	
	def reset (self):
		for key, gen in self.sub_gen.items():
			gen.reset()
	
	def generate (self):
		return {key:gen.generate() for key, gen in self.sub_gen.items()}
	
	def get_sym_obs_matrix (self):
		return {key:gen.get_sym_obs_matrix() for key, gen in self.sub_gen.items()}

class RealisticObsGenerator (ObsGen):
	def __init__ (self, state):
		self.state = state
		
		self.sub_gen_class = [	JointTarget,
								JointDelta,
								JointPos,
								# JointSpeed, # <- to remove
								
								Phase,
								RandLocalUp,
								# LocalUp,
								#RotVel, # <- to remove
								
								Cmd_PosVel,
								Cmd_RotVel,
								
								#LastAction,
								
								#Height, # <- to remove
								#LocPosVel, # <- to remove
								]
		self.sub_gen = [Gen(self.state) for Gen in self.sub_gen_class]
		
		self.obs_dim = sum([gen.obs_dim for gen in self.sub_gen])
	
	def reset (self):
		for gen in self.sub_gen:
			gen.reset()
	
	def generate (self):
		return np.concatenate([gen.generate() for gen in self.sub_gen])
	
	def get_sym_obs_matrix (self):
		to_return = np.zeros((self.obs_dim, self.obs_dim))
		a = 0
		for gen in self.sub_gen:
			b = a + gen.obs_dim
			to_return[a:b,a:b] = gen.get_sym_obs_matrix()
			a = b
		return to_return.astype(np.float32)

class SimObsGenerator (ObsGen):
	def __init__ (self, state):
		self.state = state
		
		self.sub_gen_class = [	
								# JointSpeed,
								# RotVel,
								# Height,
								# LocPosVel,
								FootFric,
								# FootClearance, 
								# FootNormal,
								MotorConsts, 
								GravityOffset,

								# # duplicate with realistic :
								# JointTarget,
								# JointDelta,
								# JointPos,
								
								# Phase,
								# # RandLocalUp,
								# LocalUp,
								# #RotVel, # <- to remove
								
								# Cmd_PosVel,
								# Cmd_RotVel,

								]
		self.sub_gen = [Gen(self.state) for Gen in self.sub_gen_class]
		
		self.obs_dim = sum([gen.obs_dim for gen in self.sub_gen])
	
	def reset (self):
		for gen in self.sub_gen:
			gen.reset()
	
	def generate (self):
		return np.concatenate([gen.generate() for gen in self.sub_gen])
	
	def get_sym_obs_matrix (self):
		to_return = np.zeros((self.obs_dim, self.obs_dim))
		a = 0
		for gen in self.sub_gen:
			b = a + gen.obs_dim
			to_return[a:b,a:b] = gen.get_sym_obs_matrix()
			a = b
		return to_return.astype(np.float32)


class VfGenerator (ObsGen):
	def __init__ (self, state):
		self.state = state
		
		self.sub_gen_class = [	
								JointSpeed,
								RotVel,
								Height,
								LocPosVel,
								FootFric,
								FootClearance, 
								FootNormal,
								MotorConsts, 
								GravityOffset,
								]
		self.sub_gen = [Gen(self.state) for Gen in self.sub_gen_class]
		
		self.obs_dim = sum([gen.obs_dim for gen in self.sub_gen])
	
	def reset (self):
		for gen in self.sub_gen:
			gen.reset()
	
	def generate (self):
		return np.concatenate([gen.generate() for gen in self.sub_gen])
	
	def get_sym_obs_matrix (self):
		to_return = np.zeros((self.obs_dim, self.obs_dim))
		a = 0
		for gen in self.sub_gen:
			b = a + gen.obs_dim
			to_return[a:b,a:b] = gen.get_sym_obs_matrix()
			a = b
		return to_return.astype(np.float32)

class MotorGenerator (ObsGen):
	def __init__ (self, state):
		self.state = state
		
		self.sub_gen_class = [	
								JointDelta,
								JointSpeed,

								Phase,
								LocalUp,

								FootClearance, 
								FootNormal,
								]
		self.sub_gen = [Gen(self.state) for Gen in self.sub_gen_class]
		
		self.obs_dim = sum([gen.obs_dim for gen in self.sub_gen])
	
	def reset (self):
		for gen in self.sub_gen:
			gen.reset()
	
	def generate (self):
		return np.concatenate([gen.generate() for gen in self.sub_gen])
	
	def get_sym_obs_matrix (self):
		to_return = np.zeros((self.obs_dim, self.obs_dim))
		a = 0
		for gen in self.sub_gen:
			b = a + gen.obs_dim
			to_return[a:b,a:b] = gen.get_sym_obs_matrix()
			a = b
		return to_return.astype(np.float32)



# -------------------------------------------- Joint related --------------------------------------------

switch_legs = np.asarray([	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], ])

class JointTarget (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
		self.mean = np.asarray([0., 0.628, -1.257] * 4)
	def generate (self):
		return np.asarray(self.state.joint_target) - self.mean
	def get_sym_obs_matrix (self):
		return switch_legs

class JointDelta (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
	def generate (self):
		return np.asarray(self.state.joint_target) - np.asarray(self.state.joint_rot)
	def get_sym_obs_matrix (self):
		return switch_legs

class ZeroJointDelta (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
	def generate (self):
		return (np.asarray(self.state.joint_target) - np.asarray(self.state.joint_rot))*0
	def get_sym_obs_matrix (self):
		return switch_legs

class JointPos (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
		self.mean = np.asarray([0., 0.628, -1.257] * 4)
	def generate (self):
		return np.asarray(self.state.joint_rot) - self.mean
	def get_sym_obs_matrix (self):
		return switch_legs

class JointSpeed (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
	def generate (self):
		return np.asarray(self.state.joint_rot_speed)/30
	def get_sym_obs_matrix (self):
		return switch_legs
	
class LastAction (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
	def generate (self):
		return np.asarray(self.state.last_action)
	def get_sym_obs_matrix (self):
		return switch_legs
	
class Phase (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 2
	def generate (self):
		return np.asarray([np.sin(self.state.phase), np.cos(self.state.phase)])
	def get_sym_obs_matrix (self):
		return np.diag([-1, -1])
		
class FootPos (ObsGen):
	def __init__(self, state):
		self.state = state
		self.obs_dim = 12
	def generate(self):
		return np.asarray(self.state.loc_foot_pos) * 10
		
class FootMeanPos (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
	def generate (self):
		return np.asarray(self.state.mean_loc_foot_pos) * 10
	
# -------------------------------------------- IMU related --------------------------------------------
class LocalUp (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def generate (self):
		return np.asarray(self.state.loc_up_vect)
	def get_sym_obs_matrix(self):
		return np.diag([1, -1, 1])

class RandLocalUp (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def reset (self):
		s = 3
		self.random_r = R.from_euler('zyx', np.random.uniform(-s,s, size=(3,)), degrees=True)
	def generate (self):
		# print(self.random_r.apply(np.asarray(self.state.loc_up_vect)))
		return self.random_r.apply(np.asarray(self.state.loc_up_vect))
		# return np.asarray([0, 0, 1])
	def get_sym_obs_matrix(self):
		return np.diag([1, -1, 1])
		
class RotVel (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def generate (self):
		return np.maximum(np.minimum(np.asarray(self.state.loc_rot_speed)*0.1, 1), -1)
	def get_sym_obs_matrix(self):
		return np.diag([-1, 1, -1])
		
# -------------------------------------------- CMD related --------------------------------------------
class Cmd_PosVel (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def generate (self):
		return np.asarray(self.state.target_speed)
		# return np.asarray([1, 0])
	def get_sym_obs_matrix(self):
		return np.diag([1, -1, 1])
		
class Cmd_RotVel (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def generate (self):
		return np.asarray(self.state.target_rot_speed)
		# return np.asarray([0])
	def get_sym_obs_matrix(self):
		return np.diag([-1, 1, -1])
		
# -------------------------------------------- True Cheating --------------------------------------------
class Height (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 1
	def generate (self):
		return np.asarray([self.state.base_pos[2]])
	def get_sym_obs_matrix(self):
		return np.diag([1])
		
class LocPosVel (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def generate (self):
		return np.asarray(self.state.loc_pos_speed)
	def get_sym_obs_matrix(self):
		return np.diag([1, -1, 1])

class FootScans (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 36
	def generate (self):
		return np.asarray(self.state.foot_scans) * 10

class FootFric (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 4
	def generate (self):
		return np.asarray(self.state.foot_f)*10
	def get_sym_obs_matrix(self):
		return np.asarray([
							[0, 1, 0, 0],
							[1, 0, 0, 0],
							[0, 0, 0, 1],
							[0, 0, 1, 0]
						])

class FootClearance (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 4
	def generate (self):
		return np.asarray(self.state.foot_clearance) * 10
	def get_sym_obs_matrix(self):
		return np.asarray([
							[0, 1, 0, 0],
							[1, 0, 0, 0],
							[0, 0, 0, 1],
							[0, 0, 1, 0]
						])

class FootNormal (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 12
	def generate (self):
		return np.asarray(self.state.loc_foot_normal.flatten())
	def get_sym_obs_matrix(self):
		return switch_legs



# -------------------------------------------- Misc --------------------------------------------

class MotorConsts (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 2
	def generate (self):
		return np.asarray([(self.state.kp0-60)/10, (self.state.kd0_fac-0.12)/0.2])
	def get_sym_obs_matrix(self):
		return np.diag([1, 1])

class GravityOffset (ObsGen):
	def __init__ (self, state):
		self.state = state
		self.obs_dim = 3
	def generate (self):
		return (self.state.loc_gravity - np.asarray([0, 0, -9.81]))
	def get_sym_obs_matrix(self):
		return np.diag([1, -1, 1])
