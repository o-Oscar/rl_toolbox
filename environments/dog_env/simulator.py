import numpy as np
import pybullet as p
import matplotlib.image as mpimg
import time
from pathlib import Path

"""
jointType : 
JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
"""

class Simulator():

	def __init__(self, state, adr, debug=False, render=False):
		# --- Step related ---
		self.state = state
		self.adr = adr
		
		self.timeStep = 1/240
		self.frameSkip = 8
		
		self.lowpass_joint_f = 15 # Hz # 15
		self.lowpass_joint_alpha = min(1, self.timeStep*self.lowpass_joint_f)
		
		self.lowpass_rew_f = 5 # Hz
		self.lowpass_rew_alpha = min(1, self.timeStep*self.lowpass_rew_f*self.frameSkip)
		
		self.curr_ep = 0
		
		# --- Render-related ---
		self.debug = debug
		self.render = render
		self.first_render = True
		self.render_path = None
		self.raw_frames = []
		self.frame = 0
		self.frame_per_render = 4
	
		# --- Connecting to the right server ---
		if self.debug:
			self.pcId = p.connect(p.GUI)
			p.resetDebugVisualizerCamera (1, 0, 0, [0, 0, 0.3])
			self.to_plot = [[] for i in range(100)]
		else:
			self.pcId = p.connect(p.DIRECT)
			
		# --- Loading the meshes ---
		urdf_path = str(Path(__file__).parent) + "/urdf"
		self.groundId = p.loadURDF(urdf_path + "/plane_001/plane.urdf", physicsClientId=self.pcId)
		self.robotId = p.loadURDF(urdf_path + "/robot_001/robot.urdf", [0,0,1], physicsClientId=self.pcId)
		
		p.setGravity(0, 0, -9.81, physicsClientId=self.pcId)
		p.setPhysicsEngineParameter(fixedTimeStep=self.timeStep, physicsClientId=self.pcId)
		
		# --- setting up the adr ---
		#self.adr.add_param("min_friction", 0.5, -0.004, 0.1)
		self.adr.add_param("min_friction", 0.5, 0, 0.1)
		
		
		friction = 0.5
		for i in range(12):
			p.changeDynamics(self.robotId, i, lateralFriction=friction, physicsClientId=self.pcId)
		p.changeDynamics(self.groundId, -1, lateralFriction=friction, physicsClientId=self.pcId)
		
	def set_epoch (self, ep):
		self.curr_ep = ep
	
	def step (self, action, joint_target):
	
		for t in range(self.frameSkip):
			self.update_joint_lowpass (joint_target)
			#self.set_help_force ()
			p.stepSimulation (physicsClientId=self.pcId)
		
			if self.render:
				self.render_frame()
			
			if self.debug:
				base_pos, base_rot = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)
				p.resetDebugVisualizerCamera (1, 30, -15, base_pos, physicsClientId=self.pcId)
				#p.resetDebugVisualizerCamera (1, 0, -15, base_pos)
		
		self.update_state(action)
		
		if self.debug:
			for i in range(12):
				self.to_plot[i].append(self.state.joint_rot[i])
				#self.to_plot[i+12].append(self.state.joint_rot_speed[i])
				self.to_plot[i+12].append(action[i])
			for i in range(4):
				self.to_plot[i+24].append(self.state.foot_clearance[i])
			for i in range(4):
				self.to_plot[i+24+8+2].append(self.state.foot_vel[i])
			self.to_plot[8+24].append(self.state.mean_planar_speed[0])
			self.to_plot[9+24].append(self.state.mean_planar_speed[1])
	"""
	def set_help_force (self):
		z = self.state.base_pos[2]
		dx = self.state.base_pos_speed[0]
		dy = self.state.base_pos_speed[1]
		dz = self.state.base_pos_speed[2]
		fac = 0 # min(max((1-self.curr_ep/500), 0), 1)
		kdz = 50*fac
		kpz = kdz*kdz*(1000/(100*100))
		h_targ = 0.27
		
		force = [-(dx-self.state.target_speed[0])*kdz, -(dy-self.state.target_speed[1])*kdz, -(z-h_targ)*kpz-dz*kdz]
		#force = [0, 0, 60*(1-self.curr_ep/1000)+20]
		
		p.applyExternalForce (self.robotId, -1, force, self.state.base_pos, p.WORLD_FRAME)
		#p.applyExternalTorque (objectUniqueId=self.robotId, linkIndex=-1, torqueObj=torque, flags=p.WORLD_FRAME)
	"""
	def update_joint_lowpass (self, joint_target):
		for i in range(12):
			self.state.joint_target[i] = self.state.joint_target[i]*(1-self.lowpass_joint_alpha) + joint_target[i]*self.lowpass_joint_alpha
			p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=self.state.joint_target[i], force=10, maxVelocity=20, physicsClientId=self.pcId)
	
	def update_state (self, action):
		# experimental
		new_pos_speed, new_rot_speed = p.getBaseVelocity(self.robotId, physicsClientId=self.pcId)
		new_pos_speed = np.asarray(new_pos_speed)
		new_rot_speed = np.asarray(new_rot_speed)
		self.state.base_pos_acc = (new_pos_speed-self.state.base_pos_speed)/(self.timeStep*self.frameSkip)
		self.state.base_rot_acc = (new_rot_speed-self.state.base_rot_speed)/(self.timeStep*self.frameSkip)
		
		# --- base pos and rot ---
		self.state.base_pos, self.state.base_rot = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)
		self.state.base_pos = list(self.state.base_pos)
		self.state.base_rot = list(self.state.base_rot)
		
		# --- base speed ---
		self.state.base_pos_speed, self.state.base_rot_speed = p.getBaseVelocity(self.robotId, physicsClientId=self.pcId)
		self.state.base_pos_speed = list(self.state.base_pos_speed)
		self.state.base_rot_speed = list(self.state.base_rot_speed)
		
		# --- joint pos and speed ---
		for i in range(p.getNumJoints(self.robotId, physicsClientId=self.pcId)):
			data = p.getJointState(self.robotId, i, physicsClientId=self.pcId)
			self.state.joint_rot[i] = data[0]
			self.state.joint_rot_speed[i] = data[1]
			self.state.joint_torque[i] = data[3]
		
		# --- body clearances ---
		all_contact_point = p.getClosestPoints(self.robotId, self.groundId, 100, linkIndexA=-1, physicsClientId=self.pcId)
		if len(all_contact_point) == 0:
			self.state.base_clearance = 0
		else:
			_, _, _, _, _, point_pos, _, _, dist, _, _, _, _, _ = all_contact_point[0]
			self.state.base_clearance = dist
		
		# --- foot clearances ---
		all_foot_id = [2, 5, 8, 11]#[1, 2, 4, 5, 7, 8, 10, 11]
		for i, link_index in enumerate(all_foot_id):
			all_contact_point = p.getClosestPoints(self.robotId, self.groundId, 100, linkIndexA=link_index, physicsClientId=self.pcId)
			if len(all_contact_point) == 0:
				self.state.foot_clearance[i] = 0
			else:
				_, _, _, _, _, point_pos, _, _, dist, _, _, _, _, _ = all_contact_point[0]
				self.state.foot_clearance[i] = dist
				
				linkWorldPosition, linkWorldOrientation, _, _, _, _, worldLinkLinearVelocity, worldLinkAngularVelocity = p.getLinkState(self.robotId, link_index, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=self.pcId)
				rel_pos = np.asarray(point_pos) - np.asarray(linkWorldPosition)
				world_vel = np.asarray(worldLinkLinearVelocity) + self.vect_prod(worldLinkAngularVelocity, rel_pos)
				self.state.foot_vel[i] = world_vel
		
		
		# --- local speed ---
		self.state.base_rot_mat = np.asarray(p.getMatrixFromQuaternion(self.state.base_rot)).reshape((3,3))
		self.state.planar_speed = np.asarray(self.state.base_pos_speed[:2])
		
		self.state.loc_up_vect = (np.asarray((0, 0, 1)).reshape((1,3)) @ self.state.base_rot_mat).flatten().tolist()
		self.state.loc_pos_speed = (np.asarray(self.state.base_pos_speed).reshape((1, 3)) @ self.state.base_rot_mat).flatten().tolist()
		self.state.loc_rot_speed = (np.asarray(self.state.base_rot_speed).reshape((1, 3)) @ self.state.base_rot_mat).flatten().tolist()
		
		# --- local vectors ---
		planar_front = np.zeros((2,))
		planar_left = np.zeros((2,))
		
		planar_front[0] = self.state.base_rot_mat[0, 0]
		planar_front[1] = self.state.base_rot_mat[1, 0]
		planar_front /= np.sqrt(np.sum(np.square(planar_front)))
		planar_left[0] = planar_front[1]
		planar_left[1] = -planar_front[0]
		
		# --- local planar speed ---
		self.state.loc_planar_speed = [np.sum(self.state.planar_speed * planar_front), np.sum(self.state.planar_speed * planar_left)]
		
		self.state.mean_planar_speed = self.state.mean_planar_speed*(1-self.lowpass_rew_alpha) + np.asarray(self.state.loc_planar_speed)*self.lowpass_rew_alpha
		self.state.mean_z_rot_speed = self.state.mean_z_rot_speed*(1-self.lowpass_rew_alpha) + self.state.base_rot_speed[2]*self.lowpass_rew_alpha
		
		for i in range(12):
			self.state.mean_joint_rot[i] = (1-self.lowpass_rew_alpha)*self.state.mean_joint_rot[i] + self.lowpass_rew_alpha*self.state.joint_rot[i]
			self.state.mean_action[i] = (1-self.lowpass_rew_alpha)*self.state.mean_action[i] + self.lowpass_rew_alpha*action[i]
		
		for i in range(12):
			self.state.acc_joint_rot[i] = (self.state.joint_rot_speed[i] - self.state.last_joint_rot_speed[i])/(self.timeStep*self.frameSkip)
			self.state.last_joint_rot_speed[i] = self.state.joint_rot_speed[i]
		
		self.state.frame += 1
		
	def reset (self, des_v, des_clear, legs_angle):
		h0 = 5
		self.state.reset()
		p.resetBasePositionAndOrientation(self.robotId, [0, 0, h0], self.state.base_rot, physicsClientId=self.pcId)
		p.resetBaseVelocity(self.robotId, self.state.base_pos_speed, self.state.base_rot_speed, physicsClientId=self.pcId)
		for i in range(12):
			self.state.joint_rot[i] = legs_angle[i]
			self.state.joint_target[i] = legs_angle[i]
			self.state.mean_joint_rot[i] = legs_angle[i]
			p.resetJointState(self.robotId, i, legs_angle[i], 0, physicsClientId=self.pcId)
		
		act_clear = self.get_clearance ()
		h = des_clear-act_clear+h0
		self.state.base_pos = [0, 0, h]
		self.state.base_pos_speed = [des_v, 0, 0]
		p.resetBasePositionAndOrientation(self.robotId, self.state.base_pos, self.state.base_rot, physicsClientId=self.pcId)
		p.resetBaseVelocity(self.robotId, self.state.base_pos_speed, self.state.base_rot_speed, physicsClientId=self.pcId)
	
		# --- adr ---
		max_friction = 1
		friction = np.random.random()*(max_friction-self.adr.value("min_friction")) + self.adr.value("min_friction")
		if self.adr.is_test_param("min_friction"):
			friction = self.adr.value("min_friction")
		
		for i in range(12):
			p.changeDynamics(self.robotId, i, lateralFriction=friction, physicsClientId=self.pcId)
		p.changeDynamics(self.groundId, -1, lateralFriction=friction, physicsClientId=self.pcId)
			
			
		#self.state.a_pos_speed = 1 - min(max(self.curr_ep/700, 0), 1)
		
	def get_clearance (self):
		to_return = 100
		all_foot_id = [1, 2, 4, 5, 7, 8, 10, 11]
		for i, link_index in enumerate(all_foot_id):
			all_contact_point = p.getClosestPoints(self.robotId, self.groundId, 100, linkIndexA=link_index, physicsClientId=self.pcId)
			if len(all_contact_point) > 0:
				_, _, _, _, _, point_pos, _, _, dist, _, _, _, _, _ = all_contact_point[0]
				to_return = min(to_return, dist)
		return to_return
	
	def vect_prod (self, a, b):
		to_return = np.empty((3,), dtype=np.float32)
		to_return[0] = a[1]*b[2] - a[2]*b[1]
		to_return[1] = a[2]*b[0] - a[0]*b[2]
		to_return[2] = a[0]*b[1] - a[1]*b[0]
		return to_return
	
	def render_frame (self):
		self.frame += 1
		if self.frame%self.frame_per_render != 0:
			return
			
		cameraTargetPosition = list(p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)[0])
		
		if self.first_render:
			self.first_render = False
			self.first_height = cameraTargetPosition[2]
			self.cam_alpha = 0
			self.cam_r = 2
		
		self.cam_alpha += np.pi*2/(30*10)
		
		base_pos = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)[0]
		cameraEyePosition = [base_pos[0]+self.cam_r*np.sin(self.cam_alpha), base_pos[1]+self.cam_r*np.cos(self.cam_alpha), 0.7]
		
		
		cameraTargetPosition[2] = self.first_height
		cameraUpVector = [0,0,1]
		width = 960
		height = 640
		
		viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVecto, physicsClientId=self.pcIdr)
		projectionMatrix = p.computeProjectionMatrixFOV(fov=40.0, aspect=width/height, nearVal=0.1, farVal=10, physicsClientId=self.pcId)
		
		width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=width,	 height=height, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, physicsClientId=self.pcId)
		self.raw_frames.append(rgbImg.reshape((width,height,-1))[:,:,:3])
		