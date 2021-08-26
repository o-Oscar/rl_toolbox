import erquy_py as erquy
import numpy as np

import os

class IdefXWorld ():
	def __init__ (self, urdf_path, mesh_path):
		self.world = erquy.World()

		self.world.loadUrdf(urdf_path, mesh_path)

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


		
	def q_to_w (self, qpos):
		return (self.q_to_w_ @ np.asarray(qpos).reshape((-1, 1))).reshape(np.asarray(qpos).shape)
	def v_to_w (self, qvel):
		return (self.v_to_w_ @  np.asarray(qvel).reshape((-1, 1))).reshape(np.asarray(qvel).shape)
	def q_to_s (self, qpos):
		return (self.q_to_w_.T @ np.asarray(qpos).reshape((-1, 1))).reshape(np.asarray(qpos).shape)
	def v_to_s (self, qvel):
		return (self.v_to_w_.T @  np.asarray(qvel).reshape((-1, 1))).reshape(np.asarray(qvel).shape)


	def setGravity (self, gravity):
		self.world.setGravity(gravity)
	
	def setTimeStep (self, timestep):
		self.world.setTimeStep(timestep)
	
	def start_vizualizer (self, urdf_path, mesh_path):
		self.viz = erquy.Visualizer(urdf_path, os.path.dirname(urdf_path))
	
	def update_vizualizer (self, qpos):
		self.viz.update(self.q_to_w(qpos))
	

	def setJointsTarget (self, joint_target):
		self.world.setPdTarget(self.q_to_w([0, 0, 0, 0, 0, 0, 1] + list(joint_target)), self.v_to_w([0]*self.world.nv()))
	
	def setJointsPdGains (self, joint_kp, joint_kd):
		n_dof = self.world.nv()
		kp = np.asarray([0] * (n_dof - 12) + list(joint_kp))
		kd = np.asarray([0] * (n_dof - 12) + list(joint_kd))
		self.world.setPdGains(kp, kd)

	def getJointTorque (self):
		generalized_torque = self.v_to_s(self.world.getPdForce())
		return generalized_torque[-12:]
	
	def setJointMaxTorque (self, max_torque):
		self.world.setMaxTorque(np.asarray([0] * 6 + list(max_torque)))
	
	def setPushTorque (self, push_torque):
		generalized_torque = list(push_torque) + [0]*3 + [0]*12
		self.world.setGeneralizedTorque(self.v_to_w(np.asarray(generalized_torque)))
	

	def getState (self):
		qpos, qvel = self.world.getState()
		return self.q_to_s(qpos), self.v_to_s(qvel)
	
	def setState(self, qpos, qvel):
		self.world.setState(self.q_to_w(qpos), self.v_to_w(qvel))
		self.world.setPdTarget(self.q_to_w(qpos), self.v_to_w(qvel))
	
	def integrate (self):
		self.world.integrate()