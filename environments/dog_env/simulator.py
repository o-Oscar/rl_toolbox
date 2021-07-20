import numpy as np
import erquy_py as erquy
import time
from pathlib import Path
import os, sys
from scipy.spatial.transform import Rotation as R
from .cmd import CMD_Setter
import matplotlib.pyplot as plt

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
		self.world = erquy.World()
		self.world.loadUrdf (urdf_path, os.path.dirname(urdf_path))

		self.world.setGravity(np.asarray([0, 0, -9.81]))
		self.world.setTimeStep(self.timeStep)
		
		
		self.joint_indexes = [i for i in range(12)]

		if self.debug:
			self.viz = erquy.Visualizer(urdf_path, os.path.dirname(urdf_path))
		
		self.footFrameIdx = [self.world.getFrameIdxByName(strid+"_foot") for strid in ["FL", "FR", "BL", "BR"]]
		
		self.jointIdx = []
		for strid in ["FL", "FR", "BL", "BR"]:
			for joint_type in ["clavicle", "arm", "forearm"]:
				self.jointIdx.append(self.world.getJointIdxByName(strid+"_"+joint_type+"_joint"))
		self.jointIdx = [id - min(self.jointIdx) for id in self.jointIdx]
		self.trunkIdx = self.world.getFrameIdxByName("trunk")
	
		joint_fac = np.asarray([1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1])

		self.q_to_w_ = np.zeros((self.world.nq(), self.world.nq()))
		self.v_to_w_ = np.zeros((self.world.nv(), self.world.nv()))
		self.q_to_w_[:7,:7] = np.identity(7)
		self.v_to_w_[:6,:6] = np.identity(6)
		for i, idx in enumerate(self.jointIdx):
			self.q_to_w_[7+idx,7+i] = joint_fac[i]
			self.v_to_w_[6+idx,6+i] = joint_fac[i]
		
		# print(list(self.world.getJointNames()))
		# exit()
		
	def q_to_w (self, qpos):
		return (self.q_to_w_ @ np.asarray(qpos).reshape((-1, 1))).reshape(np.asarray(qpos).shape)
	def v_to_w (self, qvel):
		return (self.v_to_w_ @  np.asarray(qvel).reshape((-1, 1))).reshape(np.asarray(qvel).shape)
	def q_to_s (self, qpos):
		return (self.q_to_w_.T @ np.asarray(qpos).reshape((-1, 1))).reshape(np.asarray(qpos).shape)
	def v_to_s (self, qvel):
		return (self.v_to_w_.T @  np.asarray(qvel).reshape((-1, 1))).reshape(np.asarray(qvel).shape)

	def step (self, joint_target):
		
		reset_base = "reset_base" in self.state.sim_args and self.state.sim_args["reset_base"]
		# self.world.setPdTarget(inv_qpos([0] * (self.world.nq()-12) + list(joint_target)), inv_qvel([0]*self.world.nv())) # needs a valid qpos configurationo
		self.world.setPdTarget(self.q_to_w([0, 0, 0, 0, 0, 0, 1] + list(joint_target)), self.v_to_w([0]*self.world.nv()))

		for t in range(self.frameSkip):
			current_torque = self.try_to_integrate()
			# self.world.integrate()
			if reset_base:
				qpos, qvel = self.world.getState()
				self.world.setState(np.asarray(list(self.state.sim_args["base_state"]) + list(qpos[7:])), np.asarray([0, 0, 0, 0, 0, 0] + list(qvel[6:])))
				
		# print(current_torque)
		for state in self.all_states:
			self.update_state(state, joint_target)
			
		if self.debug:
			self.viz.update(self.q_to_w(self.state.qpos))
	
	
	def try_to_integrate (self):
		n_dof = self.world.nv()
		qpos, qvel = self.world.getState()
		is_min = [False]*n_dof
		is_max = [False]*n_dof
		max_torque = [1000]*6 + [11] * 12 # 11
		current_torque = [0] * n_dof
		done = False # [(m or M) or abs(tau) < max_torque for m, M, tau in zip(is_min, is_max, current_torque)] 
		
		kp0 = [0] * n_dof # self.kp[:]
		kd0 = [0] * n_dof # self.kd[:]
		limit_torque = [0] * n_dof
		to_plot = [[] for i in range(100)]
		tests = 0
		while not np.all(done):
			for i in range(n_dof):
				if is_min[i] :
					kp0[i] = 0
					kd0[i] = 0
					limit_torque[i] = -max_torque[i]
				elif is_max[i] :
					kp0[i] = 0
					kd0[i] = 0
					limit_torque[i] = max_torque[i]
				else:
					kp0[i] = self.kp[i]
					kd0[i] = self.kd[i]
					limit_torque[i] = 0
			
			# print(limit_torque)
			self.world.setGeneralizedTorque(self.v_to_w(np.asarray(limit_torque)))
			self.world.setPdGains(np.abs(self.v_to_w(kp0)), np.abs(self.v_to_w(kd0)))
			# self.world.setPdGains(np.abs(self.v_to_w(self.kp)), np.abs(self.v_to_w(self.kd)))
			self.world.setState(qpos, qvel)
			self.world.integrate()
			
			current_torque = self.v_to_s(self.world.getPdForce())
			# print(current_torque)
			done = [(m or M) or abs(tau) < max_t for m, M, tau, max_t in zip(is_min, is_max, current_torque, max_torque)] 
			for i in range(n_dof):
				to_plot[i].append(current_torque[i])
				if current_torque[i] > max_torque[i]:
					is_max[i] = True
				elif current_torque[i] < -max_torque[i]:
					is_min[i] = True
			
			tests += 1
		
		if tests > 1 and False:
			print(tests)
			# print(is_max)
			# print(is_min)
			# print()
			"""
			for i in range(12):
				plt.plot(to_plot[i], label=str(i))
			plt.legend()
			plt.show()
			"""
		# return is_min, is_max, current_torque, max_torque

	def update_state (self, state, joint_target, update_phase=True):
	
		state.base_pos = self.world.getFramePosition(self.trunkIdx)
		state.base_r = R.from_matrix(self.world.getFrameOrientation(self.trunkIdx))
		state.base_pos_speed = self.world.getFrameVelocity(self.trunkIdx)
		state.base_rot_speed = self.world.getFrameAngularVelocity(self.trunkIdx)
		
		state.loc_pos_speed = state.base_r.inv().apply(state.base_pos_speed)
		state.loc_rot_speed = state.base_r.inv().apply(state.base_rot_speed)
		state.loc_up_vect = state.base_r.inv().apply([0, 0, 1])
		
		qpos, qvel = self.world.getState()
		qpos = self.q_to_s(qpos)
		qvel = self.v_to_s(qvel)
		calc_f = self.v_to_s(self.world.getPdForce())
		state.qpos = qpos
		state.qvel = qvel

		state.joint_rot = np.asarray(qpos[-12:])
		state.joint_target = joint_target
		state.joint_rot_speed = np.asarray(qvel[-12:])
		state.joint_torque = np.asarray(calc_f[-12:])
		
		state.foot_force = self.get_feet_force()

		for i, idx in enumerate(self.footFrameIdx):
			state.foot_pos[i] = self.world.getFramePosition(idx)
			state.foot_speed[i] = self.world.getFrameVelocity(idx)
			state.loc_foot_pos[i] = state.base_r.inv().apply(state.foot_pos[i] - state.base_pos)
			last_loc_foot_speed = state.loc_foot_speed[i]
			state.loc_foot_speed[i] = state.foot_speed[i] - state.base_pos_speed - np.cross(state.base_rot_speed, state.foot_pos[i] - state.base_pos)
			state.loc_foot_speed[i] = state.base_r.inv().apply(state.loc_foot_speed[i])
			state.loc_foot_acc[i] = (state.loc_foot_speed[i]-last_loc_foot_speed)/(self.timeStep*self.frameSkip)
		
		if update_phase:
			state.frame += 1
			if not "update_phase" in self.state.sim_args or self.state.sim_args["update_phase"]:
				state.phase += 2*np.pi*state.f0*self.timeStep*self.frameSkip
			
			if not "update_cmd" in self.state.sim_args or self.state.sim_args["update_cmd"]:
				self.cmdSetter.update_cmd()
		
	def get_feet_force (self):
		to_return = [np.zeros((3,)) for i in range(4)]
		joint_interest = [-1, -1] + [-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3]
		n_contact, all_joint_ids, forces = self.world.getContactInfos()
		all_joint_ids = all_joint_ids.reshape((-1,2))
		for i in range(n_contact):
			for joint_id in all_joint_ids[i]:
				if joint_interest[joint_id] > -1:
					to_return[joint_interest[joint_id]] = to_return[joint_interest[joint_id]] + forces[i*3:i*3+3]
		return to_return
			
	def reset (self, render=True):

		# ---- setting pids ----
		n_dof = self.world.nv()
		kp0 = self.state.sim_args["kp"] if "kp" in self.state.sim_args else 72 # <- standard 72, best according to a simple test : 60
		kd0 = kp0* (self.state.sim_args["kd_fac"] if "kd_fac" in self.state.sim_args else 0.1) # <- standard
		self.kp0 = kp0
		self.kd0 = kd0
		# kp0 = 62
		# kd0 = kp0*0.1 
		
		r = 24/30
		kp = np.asarray([0] * (n_dof - 12) + [kp0, kp0, kp0*r]*4)
		kd = np.asarray([0] * (n_dof - 12) + [kd0, kd0, kd0*r]*4)
		self.kp = kp
		self.kd = kd
		self.world.setPdGains(kp, kd)
		
		
		cur_frame = 0
		cur_phase = self.state.sim_args["phase"] if "phase" in self.state.sim_args else np.random.random() * 2 * np.pi
		
		if "base_state" in self.state.sim_args and "leg_pos" in self.state.sim_args:
			# cur_phase = 2*np.pi/30
			qvel = [0]*self.world.nv()
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
		
		legs_angle = np.asarray(qpos[-12:])
		self.world.setPdTarget(self.q_to_w(qpos), self.v_to_w(qvel))
		self.world.setState(self.q_to_w(qpos), self.v_to_w(qvel))
		
		self.world.setMaterialPairProp(-1, -1, 0.5)

		self.max_f = 0.1
		self.min_f = 0.
		# self.all_f = [np.random.uniform(self.min_f, self.max_f) for i in range(4)]
		# self.all_f = [0, 0.5, 0.5, 0.5]
		f = np.random.uniform(self.min_f, self.max_f)
		# self.state.foot_f = [np.random.uniform(self.min_f, self.max_f) for i in range(4)]
		self.state.foot_f = [f for i in range(4)]
		if "foot_f" in self.state.sim_args:
			self.state.foot_f = self.state.sim_args["foot_f"]
		for i, strid in enumerate(["FL", "FR", "BL", "BR"]):
			self.world.setMaterialPairProp(self.world.getJointIdxByName("universe"), self.world.getJointIdxByName(strid+"_forearm_joint"), self.state.foot_f[i])
		

		# --- actual state reset ---
		for state in self.all_states:
			state.reset(frame=cur_frame, phase=cur_phase)
			self.update_state(state, legs_angle, update_phase=False)
		
		self.cmdSetter.reset_cmd(targ_vel)
		if self.debug and render:
			self.viz.update(self.q_to_w(self.state.qpos))

	def close (self):
		pass
	