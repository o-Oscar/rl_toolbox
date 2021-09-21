import erquy_py as erquy
import numpy as np

import os

class IdefXWorld ():
	def __init__ (self, urdf_path, mesh_path, use_flex=False):
		self.world = erquy.World()
		self.use_flex = use_flex

		self.world.loadUrdf(urdf_path, mesh_path)

		self.nq = self.world.nq()
		self.nv = self.world.nv()

		self.footFrameIdx = [self.world.getFrameIdxByName(strid+"_foot") for strid in ["FL", "FR", "BL", "BR"]]
		
		# print("trunk_joint :", self.world.getJointIdxByName("trunk_joint"))
		# self.trunkIdx = self.world.getFrameIdxByName("trunk")
		# print(self.world.getJointGeneralizedPosition(self.world.getJointIdxByName("trunk_joint")))


		self.jointNames = []
		for strid in ["FL", "FR", "BL", "BR"]:
			for joint_type in ["clavicle", "arm", "forearm"]:
				self.jointNames.append(strid+"_"+joint_type+"_joint")

		self.jointIdx = [self.world.getJointIdxByName(name) for name in self.jointNames]
		self.jointGeneralizedIdx = [self.world.getJointGeneralizedPosition(idx) for idx in self.jointIdx]
		self.joint_facs = np.asarray([1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1])


		self.flexNames = []
		if self.use_flex:
			for strid in ["FL", "FR", "BL", "BR"]:
				for joint_type in ["clavicle", "arm", "forearm"]:
					self.flexNames.append(strid+"_"+joint_type+"_inbetween_joint")

		self.flexIdx = [self.world.getJointIdxByName(name) for name in self.flexNames]
		self.flexGeneralizedIdx = [self.world.getJointGeneralizedPosition(idx) for idx in self.flexIdx]


	def setGravity (self, gravity):
		self.world.setGravity(np.asarray(gravity))
	
	def setTimeStep (self, timestep):
		self.world.setTimeStep(timestep)
	
	def start_vizualizer (self, urdf_path, mesh_path):
		self.viz = erquy.Visualizer(urdf_path, os.path.dirname(urdf_path))
	
	def update_vizualizer (self):
		qpos, qvel = self.world.getState()
		self.viz.update(qpos)
	

	def setJointsTarget (self, joint_target):
		q_target = np.zeros((self.nq,))
		v_target = np.zeros((self.nv,))

		for i in range(12):
			q_idx, v_idx, _, _ = self.jointGeneralizedIdx[i]
			q_target[q_idx] = joint_target[i]*self.joint_facs[i]
		
		q_target[6] = 1 # to make the quaternion of the trunk happy
		self.world.setPdTarget(q_target, v_target)
	
	def setJointsPdGains (self, joint_kp, joint_kd):
		world_kp = np.zeros((self.nv,))
		world_kd = np.zeros((self.nv,))
		for i in range(12):
			q_idx, v_idx, _, _ = self.jointGeneralizedIdx[i]
			world_kp[v_idx] = joint_kp[i]
			world_kd[v_idx] = joint_kd[i]
		for i in range(len(self.flexNames)):
			q_idx, v_idx, _, _ = self.flexGeneralizedIdx[i]
			world_kp[v_idx] = 600
			world_kd[v_idx] = world_kp[v_idx]/20
		self.world.setPdGains(world_kp, world_kd)


	def getJointTorque (self):
		generalized_torque = self.world.getPdForce()
		to_return = np.zeros((12,))
		for i in range(12):
			q_idx, v_idx, _, _ = self.jointGeneralizedIdx[i]
			to_return[i] = generalized_torque[v_idx]*self.joint_facs[i]
		return to_return
	
	def setJointMaxTorque (self, max_torque):
		world_max_torque = np.zeros((self.nv,))
		for i in range(12):
			q_idx, v_idx, _, _ = self.jointGeneralizedIdx[i]
			world_max_torque[v_idx] = max_torque[i]
		for i in range(len(self.flexNames)):
			q_idx, v_idx, _, _ = self.flexGeneralizedIdx[i]
			world_max_torque[v_idx] = 10000000
		self.world.setMaxTorque(world_max_torque)
	
	def setPushTorque (self, push_torque):
		generalized_torque = np.zeros((self.nv, ))
		generalized_torque[:3] = push_torque
		self.world.setGeneralizedTorque(np.asarray(generalized_torque))
	

	def integrate (self):
		self.world.integrate()



	def setState (self, base_q, joint_pos, base_v=np.zeros((6,)), joint_vel=np.zeros((12,))):
		qpos = np.zeros((self.nq,))
		qvel = np.zeros((self.nv,))
		for i in range(12):
			q_idx, v_idx, _, _ = self.jointGeneralizedIdx[i]
			qpos[q_idx] = joint_pos[i]*self.joint_facs[i]
			qvel[v_idx] = joint_vel[i]*self.joint_facs[i]
		qpos[0:7] = base_q
		qvel[0:6] = base_v
		self.world.setState(qpos, qvel)
		self.world.setPdTarget(qpos, qvel)

	def getJointState (self):
		qpos, qvel = self.world.getState()
		joint_rot = np.zeros((12,))
		joint_vel = np.zeros((12,))
		for i in range(12):
			q_idx, v_idx, _, _ = self.jointGeneralizedIdx[i]
			joint_rot[i] = qpos[q_idx]*self.joint_facs[i]
			joint_vel[i] = qvel[v_idx]*self.joint_facs[i]
		return joint_rot, joint_vel
	
	# def getState (self):
	# 	qpos, qvel = self.world.getState()
	# 	return self.q_to_s(qpos), self.v_to_s(qvel)
	
	# def setState(self, qpos, qvel):
	# 	self.world.setState(self.q_to_w(qpos), self.v_to_w(qvel))
	# 	self.world.setPdTarget(self.q_to_w(qpos), self.v_to_w(qvel))
	