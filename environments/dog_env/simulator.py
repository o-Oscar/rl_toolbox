import numpy as np
import time
from pathlib import Path
import os, sys
from scipy.spatial.transform import Rotation as R
from .cmd import CMD_Setter
import matplotlib.pyplot as plt
from .erquy_interface import IdefXWorld

# TODO : faire deux versions d'IdefX : Une avec terrain accidenté, l'autre avec terrain plat (actuel)
# TODO : récupérer les poids de la dernière couche du réseau entraîné par PPO pour les transférer vers le student.

class Simulator():

	def __init__(self, state, debug=False):
		# --- Step related ---
		self.state = state
		self.all_states = [self.state]
		self.cmdSetter = CMD_Setter (self.all_states, False)
		
		self.frameSkip = 10
		self.timeStep = 1/(30*self.frameSkip)
		
		# --- Render-related ---
		self.debug = debug
	
		# --- creating the simulation World ---
		urdf_path = os.path.join("data", "idefX", "idefX.urdf")
		# urdf_path = os.path.join("data", "idefX", "idefX_flex.urdf")
		# urdf_path = os.path.join("data", "idefX_valley", "idefX.urdf")
		self.world = IdefXWorld(urdf_path, os.path.dirname(urdf_path))

		self.world.setGravity(np.asarray([0, 0, -9.81]))
		self.world.setTimeStep(self.timeStep)
		
		
		self.joint_indexes = [i for i in range(12)]

		if self.debug:
			# self.viz = erquy.Visualizer(urdf_path, os.path.dirname(urdf_path))
			self.world.start_vizualizer(urdf_path, os.path.dirname(urdf_path))
		
		self.footFrameIdx = [self.world.world.getFrameIdxByName(strid+"_foot") for strid in ["FL", "FR", "BL", "BR"]]
		
		self.jointIdx = []
		for strid in ["FL", "FR", "BL", "BR"]:
			for joint_type in ["clavicle", "arm", "forearm"]:
				self.jointIdx.append(self.world.world.getJointIdxByName(strid+"_"+joint_type+"_joint"))
		self.jointIdx = [id - min(self.jointIdx) for id in self.jointIdx]
		self.trunkIdx = self.world.world.getFrameIdxByName("trunk")
	
		joint_fac = np.asarray([1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1])

		self.q_to_w_ = np.zeros((self.world.world.nq(), self.world.world.nq()))
		self.v_to_w_ = np.zeros((self.world.world.nv(), self.world.world.nv()))
		self.q_to_w_[:7,:7] = np.identity(7)
		self.v_to_w_[:6,:6] = np.identity(6)
		for i, idx in enumerate(self.jointIdx):
			self.q_to_w_[7+idx,7+i] = joint_fac[i]
			self.v_to_w_[6+idx,6+i] = joint_fac[i]
		
		# print(list(self.world.getJointNames()))
		# exit()
		self.all_forces = []

		
	def q_to_w (self, qpos):
		return (self.q_to_w_ @ np.asarray(qpos).reshape((-1, 1))).reshape(np.asarray(qpos).shape)
	def v_to_w (self, qvel):
		return (self.v_to_w_ @  np.asarray(qvel).reshape((-1, 1))).reshape(np.asarray(qvel).shape)
	def q_to_s (self, qpos):
		return (self.q_to_w_.T @ np.asarray(qpos).reshape((-1, 1))).reshape(np.asarray(qpos).shape)
	def v_to_s (self, qvel):
		return (self.v_to_w_.T @  np.asarray(qvel).reshape((-1, 1))).reshape(np.asarray(qvel).shape)

	def step (self, joint_target, action):
		
		reset_base = "reset_base" in self.state.sim_args and self.state.sim_args["reset_base"]
		# self.world.setPdTarget(self.q_to_w([0, 0, 0, 0, 0, 0, 1] + list(joint_target+self.state.random_joint_delta)), self.v_to_w([0]*self.world.nv()))
		# self.world.setMaxTorque(np.asarray([0] * 6 + [11, 11, 11*24/30] * 4))
		self.world.setJointsTarget(joint_target+self.state.random_joint_delta)
		self.world.setJointMaxTorque(np.asarray([11, 11, 11*24/30] * 4))

		# push handling
		# self.push_N = 60
		# self.push_f = 50
		# self.push_n = 2
		if (self.state.frame-self.push_dN)%self.push_N < self.push_n:
			theta = np.random.random()*2*np.pi
			cur_push_f_x = self.push_f*np.sin(theta)
			cur_push_f_y = self.push_f*np.cos(theta)
		else:
			cur_push_f_x = 0
			cur_push_f_y = 0
		push_torque = [cur_push_f_x, cur_push_f_y, 0]# + [0]*3 + [0]*12
		# self.world.setGeneralizedTorque(self.v_to_w(np.asarray(push_torque)))
		self.world.setPushTorque(push_torque)
		# self.world.setPushTorque([0, 0, 0])

		for t in range(self.frameSkip):
			# current_torque = self.try_to_integrate()
			self.world.integrate()
			
			# self.all_forces.append(self.get_feet_force())

			if reset_base:
				joint_pos, joint_vel = self.world.getJointState()
				self.world.setState(self.state.sim_args["base_state"], joint_pos, joint_vel=joint_vel)
			for state in self.all_states:
				if t == state.update_t:
					self.update_state(state, joint_target, action=action)
			# print(self.get_feet_force())
			# time.sleep(0.03)
			# self.viz.update(self.world.getState()[0])

		# print(current_torque)
		# for state in self.all_states:
		# 	self.update_state(state, joint_target, action=action)
			
		if self.debug:
			# self.viz.update(self.q_to_w(self.state.qpos))
			self.world.update_vizualizer()

	def update_state (self, state, joint_target, update_phase=True, action=np.zeros((12,))):
	
		state.base_pos = self.world.world.getFramePosition(self.trunkIdx)
		state.base_r = R.from_matrix(self.world.world.getFrameOrientation(self.trunkIdx))
		state.base_pos_speed = self.world.world.getFrameVelocity(self.trunkIdx)
		state.base_rot_speed = self.world.world.getFrameAngularVelocity(self.trunkIdx)
		
		state.loc_pos_speed = state.base_r.inv().apply(state.base_pos_speed)
		state.loc_rot_speed = state.base_r.inv().apply(state.base_rot_speed)
		state.loc_up_vect = state.base_r.inv().apply([0, 0, 1])
		
		world_joint_rot, world_joint_vel = self.world.getJointState()
		# state.qpos = qpos
		# state.qvel = qvel

		state.joint_rot = world_joint_rot - self.state.random_joint_delta
		state.joint_target = joint_target
		state.joint_rot_speed = world_joint_vel
		state.joint_torque = self.world.getJointTorque()
		
		state.foot_force = self.get_feet_force()
		state.foot_clearance, state.foot_normal = self.get_feet_clearance()

		for i, idx in enumerate(self.footFrameIdx):
			state.foot_pos[i] = self.world.world.getFramePosition(idx)
			state.foot_speed[i] = self.world.world.getFrameVelocity(idx)
			state.loc_foot_pos[i] = state.base_r.inv().apply(state.foot_pos[i] - state.base_pos)
			last_loc_foot_speed = state.loc_foot_speed[i]
			state.loc_foot_speed[i] = state.foot_speed[i] - state.base_pos_speed - np.cross(state.base_rot_speed, state.foot_pos[i] - state.base_pos)
			state.loc_foot_speed[i] = state.base_r.inv().apply(state.loc_foot_speed[i])
			state.loc_foot_acc[i] = (state.loc_foot_speed[i]-last_loc_foot_speed)/(self.timeStep*self.frameSkip)
			state.loc_foot_normal[i] = state.base_r.inv().apply(state.foot_normal[i])
			if i%2 == 1:
				state.loc_foot_normal[i][1] = -state.loc_foot_normal[i][1]
		
		state.loc_gravity = state.base_r.inv().apply(state.gravity)
		state.last_action = action
		
		if update_phase:
			state.frame += 1
			if not "update_phase" in self.state.sim_args or self.state.sim_args["update_phase"]:
				state.phase += 2*np.pi*state.f0*self.timeStep*self.frameSkip
			
			if not "update_cmd" in self.state.sim_args or self.state.sim_args["update_cmd"]:
				self.cmdSetter.update_cmd()
		
	def get_feet_force (self): # BrOcKeN
		to_return = [np.zeros((3,)) for i in range(4)]
		joint_interest = [-1]*6 + [-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3] + [-1] * 10
		n_contact, all_joint_ids, forces = self.world.world.getContactInfos()
		all_joint_ids = all_joint_ids.reshape((-1,2))
		for i in range(n_contact):
			for joint_id in all_joint_ids[i]:
				if joint_interest[joint_id] > -1:
					to_return[joint_interest[joint_id]] = to_return[joint_interest[joint_id]] + forces[i*3:i*3+3]
		return to_return
	
	def get_feet_clearance (self): # this function is ABSOLUTELY NOT urdf independant. Refacto this as soon as someone understands pinocchio better
		all_dist = self.world.world.computeDistances()
		feet_dist = all_dist[:4]
		pos0 = feet_dist[:,:3]
		pos1 = feet_dist[:,3:6]
		normal = feet_dist[:,6:]
		clearance = np.sqrt(np.sum(np.square(pos0-pos1), axis=1))
		ret_normal = np.zeros_like(normal)
		ret_clearance = np.zeros_like(clearance)
		for i, idx in enumerate([2, 3, 0, 1]):
			if np.any(np.isnan(normal[i])):
				normal[i] = pos0-pos1
				normal[i] = normal[i]/np.sqrt(np.sum(np.square(normal[i])))
			if normal[i,2] < 0:
				normal[i] = -normal[i]
			ret_normal[idx] = normal[i]
			ret_clearance[idx] = clearance[i]
		return ret_clearance, ret_normal

	def reset (self, render=True):

		s = 3
		self.state.random_joint_delta = np.random.uniform(-np.pi/180*s, np.pi/180*s, size=(12,))
		
		self.push_N = 60
		self.push_dN = self.state.sim_args["push_dN"] if "push_dN" in self.state.sim_args else 30
		self.push_f = self.state.sim_args["push_f"] if "push_f" in self.state.sim_args else 250
		self.push_n = 1

		# ---- setting pids ----

		kp0 = 60 + np.random.uniform(-10, 0) # 30 ?
		kd0_fac = 0.05 + np.random.uniform(-0.01, 0.01)
		# kd0_fac = 0.05 + np.random.uniform(-0.02, 0.02)

		kp0 = self.state.sim_args["kp"] if "kp" in self.state.sim_args else kp0 # <- standard 72, best according to a simple test : 60
		kd0_fac = (self.state.sim_args["kd_fac"] if "kd_fac" in self.state.sim_args else kd0_fac) # <- standard
		kd0 = kp0 * kd0_fac
		self.kp0 = kp0
		self.kd0 = kd0
		self.state.kp0 = kp0
		self.state.kd0_fac = kd0_fac
		# kp0 = 62
		# kd0 = kp0*0.1 
		
		r = 24/30
		kp = np.asarray([kp0, kp0, kp0*r]*4) # + np.random.uniform(-5, 5, size=12)
		kd = [kd0, kd0, kd0*r]*4
		self.kp = kp
		self.kd = kd
		self.world.setJointsPdGains(kp, kd)
		
		
		cur_frame = 0
		cur_phase = self.state.sim_args["phase"] if "phase" in self.state.sim_args else np.random.random() * 2 * np.pi
		
		if "base_state" in self.state.sim_args and "leg_pos" in self.state.sim_args:
			# cur_phase = 2*np.pi/30
			qvel = [0]*self.world.world.nv()
			qpos = list(self.state.sim_args["base_state"]) + list(self.state.sim_args["leg_pos"])
			targ_vel = [0.4, 0, 0]
		elif "ref_name" in self.state.sim_args:
			qpos, qvel = self.state.reference_bag.get_ref(self.state.sim_args["ref_name"], cur_phase)
			targ_vel = qvel[0:3]
		else:
			qpos, qvel = self.state.reference_bag.get_random_ref(cur_phase)
			targ_vel = qvel[0:3]
		targ_vel = [0.4, 0, 0]

		if "base_state" in self.state.sim_args:
			qpos[:7] = list(self.state.sim_args["base_state"])
		
		qpos[3:7] = list(np.asarray(qpos[3:7])/np.sqrt(np.sum(np.square(qpos[3:7]))))

		legs_angle = np.asarray(qpos[-12:])
		# self.world.setPdTarget(self.q_to_w(qpos), self.v_to_w(qvel))
		# self.world.setState(self.q_to_w(qpos), self.v_to_w(qvel))
		self.world.setState(qpos[:7], qpos[7:])
		
		"""
		self.max_f = 0.1
		self.min_f = 0.
		# self.all_f = [np.random.uniform(self.min_f, self.max_f) for i in range(4)]
		# self.all_f = [0, 0.5, 0.5, 0.5]
		f = np.random.uniform(self.min_f, self.max_f)
		# self.state.foot_f = [np.random.uniform(self.min_f, self.max_f) for i in range(4)]
		self.state.foot_f = [f for i in range(4)]
		"""
		r = np.random.random()
		if r < 1/3:
			f0 = 0.1
		elif r < 2/3:
			f0 = 0.2
		else:
			f0 = 0.3

		foot_f = np.maximum(0, np.random.uniform(f0-0.05, f0+0.05, size=4))

		self.world.world.setMaterialPairProp(-1, -1, 0.5)

		if "foot_f" in self.state.sim_args:
			foot_f = self.state.sim_args["foot_f"]
		for i, strid in enumerate(["FL", "FR", "BL", "BR"]):
			self.world.world.setMaterialPairProp(self.world.world.getJointIdxByName("universe"), self.world.world.getJointIdxByName(strid+"_forearm_joint"), foot_f[i])
		
		
		if "gravity" in self.state.sim_args:
			gravity = np.asarray(self.state.sim_args["gravity"])
		else:
			gravity = [0, np.random.uniform(-0.5, 0.5), -9.81]
		self.world.setGravity(gravity)


		# --- actual state reset ---
		for state in self.all_states:
			state.reset(frame=cur_frame, phase=cur_phase)
			state.kp0 = kp0
			state.kd0_fac = kd0_fac
			state.gravity = gravity
			state.foot_f = foot_f
			self.update_state(state, legs_angle, update_phase=False)
		
		self.cmdSetter.reset_cmd(targ_vel)
		if self.debug and render:
			# self.viz.update(self.q_to_w(self.state.qpos))
			self.world.update_vizualizer()

	def close (self):
		pass
	