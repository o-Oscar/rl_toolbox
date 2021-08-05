import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os
from .reference import ReferenceBag

class State:
	def __init__ (self):
		self.reset()
		self.reference_bag = ReferenceBag()
		self.sim_args = {}
		
	def reset (self, phase=0, frame=0):
		self.qpos = [0] * 19
		self.qvel = [0] * 18

		self.base_pos = [0, 0, 0]
		self.base_r = 0
		self.base_pos_speed = [0, 0, 0]
		self.base_rot_speed = [0, 0, 0]
		
		self.joint_rot = [0]*12
		self.joint_target = [0]*12
		self.joint_rot_speed = [0]*12
		self.joint_torque = [0]*12
		
		self.foot_force = np.zeros((4,3))
		self.foot_pos = [[0, 0, 0] for i in range(4)]
		self.foot_speed = [[0, 0, 0] for i in range(4)]
		self.foot_f = [0 for i in range(4)]
		self.loc_foot_pos = [[0, 0, 0] for i in range(4)]
		self.loc_foot_speed = [[0, 0, 0] for i in range(4)]
		self.loc_foot_acc = [[0, 0, 0] for i in range(4)]
		self.foot_clearance = np.zeros((4,))
		self.foot_normal = np.zeros((4,3))
		self.loc_foot_normal = np.zeros((4,3))
		self.other_contact = False

		self.gravity = np.asarray([0, 0, -9.81])
		self.loc_gravity = np.asarray([0, 0, -9.81])
		self.kp0 = 60
		self.kd0_fac = 0.12

		self.last_action = np.zeros((12,))

		self.phase = phase
		self.frame = frame
		self.f0 = 1.
		# self.f0 = 1.5
		
		
		self.loc_pos_speed = [0, 0, 0]
		self.loc_rot_speed = [0, 0, 0]
		self.loc_up_vect = [0, 0, 1]
	
		self.target_speed = np.asarray([1, 0, 0])*1
		self.target_rot_speed = np.asarray([0, 0, 0])
		
		# self.friction_f = 0.3