import numpy as np
from pathlib import Path

import config


class DogState:
	def __init__ (self, adr):
		self.adr = adr
		
		self.obs_mean = []
		self.obs_std = []
		for state_Id in config.state_vect:
			if state_Id == config.JOINT_POS or state_Id == config.JOINT_POS_RAND:
				self.obs_mean += [0.0, 1.0408382989215212, -1.968988857605835]*4
				self.obs_std += [0.2]*12
			elif state_Id == config.JOINT_VEL or state_Id == config.JOINT_VEL_RAND:
				self.obs_mean += [0.0, 1.0408382989215212, -1.968988857605835]*4
				self.obs_std += [0.2]*12
			elif state_Id == config.LOCAL_UP or state_Id == config.LOCAL_UP_RAND:
				self.obs_mean += [0, 0, 1]
				self.obs_std += [0.1]*3
			elif state_Id == config.ROT_VEL or state_Id == config.ROT_VEL_RAND:
				self.obs_mean += [0]*3
				self.obs_std += [2]*3
				
			elif state_Id == config.POS_VEL_CMD:
				self.obs_mean += [0]*2
				self.obs_std += [1]*2
			elif state_Id == config.ROT_VEL_CMD:
				self.obs_mean += [0]
				self.obs_std += [1]
				
			elif state_Id == config.HEIGHT:
				self.obs_mean += [0.2]
				self.obs_std += [0.02]
			elif state_Id == config.POS_VEL:
				self.obs_mean += [0]*3
				self.obs_std += [0.2]*3
			elif state_Id == config.MEAN_POS_VEL:
				self.obs_mean += [0]*2
				self.obs_std += [0.2]*2
			elif state_Id == config.MEAN_ROT_VEL:
				self.obs_mean += [0]
				self.obs_std += [2]
			elif state_Id == config.ACT_OFFSET:
				self.obs_mean += [0] * 12
				self.obs_std += [1] * 12
			else:
				print("ERROR : invalid obs config")
				print(1/0)
		
		# --- adr ---
		
		for name in ["joint_pos", "local_up", "rot_vect"]:
			#self.adr.add_param(name+"_rand_offset", 0, 0.003, 1000)
			#self.adr.add_param(name+"_rand_std", 0, 0.003, 1000)
			self.adr.add_param(name+"_rand_offset", 0, 0, 1000)
			self.adr.add_param(name+"_rand_std", 0, 0, 1000)
		
		self.reset()
		
		
	def reset (self):
		self.base_pos = [0, 0, 0.25]
		self.base_rot = [0, 0, 0, 1]
		self.joint_rot = [0]*12
		self.joint_target = [0]*12
		
		self.base_pos_speed = [0, 0, 0]
		self.base_rot_speed = [0, 0, 0]
		self.joint_rot_speed = [0]*12
		
		self.base_clearance = 0
		self.foot_clearance = [0]*4
		self.foot_vel = [0]*4
		
		
		self.base_rot_mat = np.identity(3)
		self.planar_speed = [0, 0]
		self.loc_planar_speed = [0, 0]
		self.loc_up_vect = [0, 0, 1]
		self.loc_pos_speed = [0, 0, 0]
		self.loc_rot_speed = [0, 0, 0]
		
		self.mean_planar_speed = np.asarray([0, 0])
		self.mean_z_rot_speed = 0
		self.mean_joint_rot = [0]*12
		self.mean_action = [0.5]*12
		self.target_pose = np.asarray([0.5, 0.5, 0.3] * 4)
		
		self.acc_joint_rot = [0]*12
		self.last_joint_rot_speed = [0]*12
	
		self.target_speed = np.asarray([1, 0])*1
		self.target_rot_speed = 0
		
		self.frame = 0
		
		# experimental
		self.joint_torque = [0]*12
		#self.contact_force = [0]*4
		self.base_pos_acc = [0, 0, 0]
		self.base_rot_acc = [0, 0, 0]
		self.act_offset = np.zeros((12,))
		self.joint_offset = np.zeros((12,))
		self.loc_up_vect_offset = np.zeros((3,))
		
		self.joint_fac = [1]*12
		
		# --- adr ---
		self.rand_offset = []
		for name, size in [("joint_pos", (12,)), ("local_up", (3,)), ("rot_vect", (3,))]:
			self.rand_offset.append((np.random.random(size=size)*2-1)*self.adr.value(name+"_rand_offset"))
			
		
	# height (1D) + local_up_vect(3D) + loc_pos_speed(3D) + loc_rot_speed(3D) + joint_angle(12D) # + joint_rot_speed(12D)
	def calc_obs (self):
		
		rand_delta = []
		for i, (name, size) in enumerate([("joint_pos", (12,)), ("local_up", (3,)), ("rot_vect", (3,))]):
			rand_delta.append(self.rand_offset[i] + np.random.normal(size=size) * self.adr.value(name+"_rand_std"))
		delta_joint_pos, delta_local_up, delta_rot_vect = rand_delta
		
		
		to_return = []
		for state_Id in config.state_vect:
		
			if state_Id == config.JOINT_POS:
				for i in range(12):
					to_return.append(self.joint_rot[i]*self.joint_fac[i] + self.joint_offset[i])
			elif state_Id == config.JOINT_POS_RAND:
				for i in range(12):
					to_return.append(self.joint_rot[i]*self.joint_fac[i] + delta_joint_pos[i])
					
			elif state_Id == config.JOINT_VEL:
				for i in range(12):
					to_return.append(self.joint_rot_speed[i]*self.joint_fac[i])
			elif state_Id == config.JOINT_VEL_RAND:
				for i in range(12):
					to_return.append(self.joint_rot_speed[i]*self.joint_fac[i])
					
			elif state_Id == config.LOCAL_UP:
				#to_return += [self.loc_up_vect[0], self.loc_up_vect[1], self.loc_up_vect[2]]
				to_return += [0, 0, 1]
			elif state_Id == config.LOCAL_UP_RAND:
				to_return += [self.loc_up_vect[0]+self.loc_up_vect_offset[0], self.loc_up_vect[1]+self.loc_up_vect_offset[1], self.loc_up_vect[2]+self.loc_up_vect_offset[2]]
				
			elif state_Id == config.ROT_VEL:
				to_return += [self.loc_rot_speed[0], self.loc_rot_speed[1], self.loc_rot_speed[2]]
			elif state_Id == config.ROT_VEL_RAND:
				to_return += [self.loc_rot_speed[0]+delta_rot_vect[0], self.loc_rot_speed[1]+delta_rot_vect[1], self.loc_rot_speed[2]+delta_rot_vect[2]]
				
			elif state_Id == config.POS_VEL_CMD:
				to_return += [self.target_speed[0], self.target_speed[1]]
			elif state_Id == config.ROT_VEL_CMD:
				to_return += [self.target_rot_speed]
				
			elif state_Id == config.HEIGHT:
				to_return += [self.base_pos[2]]
			elif state_Id == config.POS_VEL:
				to_return += [self.loc_pos_speed[0], self.loc_pos_speed[1], self.loc_pos_speed[2]]
			elif state_Id == config.MEAN_POS_VEL:
				to_return += [self.mean_planar_speed[0], self.mean_planar_speed[1]]
			elif state_Id == config.MEAN_ROT_VEL:
				to_return += [self.mean_z_rot_speed]
			elif state_Id == config.ACT_OFFSET:
				for i in range(12):
					to_return += [self.act_offset[i]]
			else:
				print("ERROR : invalid obs config")
				print(1/0)
				
		"""
		for i in range(12):
			to_return.append(self.joint_rot_speed[i])
		"""
		
		return [np.asarray(to_return)]
		
		