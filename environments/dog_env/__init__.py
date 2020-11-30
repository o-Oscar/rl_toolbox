
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path


from .dogState import DogState
from .kinematics import Kinematics
from .simulator import Simulator
from .reward import NotDyingsRew, SpeedRew, TurningRew, PoseRew, JointAccRew, BaseAccRew, FootClearRew, BaseRotRew, BaseClearRew

from .symetry import Symetry
from .blindfold import Blindfold
from .adr import Adr

import config

class DogEnv():
	def __init__(self, debug=False, render=False):
		self.symetry = Symetry()
		self.blindfold = Blindfold()
		
		self.debug = debug
		self.render = render
		
		self.adr = Adr()
		self.test_adr = False
		self.adr_rollout_len = 400
		self.frame_at_speed = 0
		
		self.kin = Kinematics()
		self.state = DogState(self.adr)
		self.sim = Simulator(self.state, self.adr, self.debug, self.render)
		# not dying - moving at the right speed - turning at the right speed - keeping joint acceleration low - keeping body acceleration low
		self.rewards = [NotDyingsRew(self.state), 
						SpeedRew(self.state), 
						TurningRew(self.state), 
						PoseRew(self.state), 
						JointAccRew(self.state), 
						BaseAccRew(self.state),
						BaseRotRew(self.state),
						FootClearRew(self.state),
						BaseClearRew(self.state)]
		
		self.curr_ep = 0
		
		
		# multiple obs and act in one obs
		self.obs_pool = []
		self.pool_len = config.obs_transition_len
		#self.obs_mean = np.concatenate([self.state.obs_mean + [0.5 for j in range(12)] for i in range(self.pool_len)])
		#self.obs_std = np.concatenate([self.state.obs_std + [1 for j in range(12)] for i in range(self.pool_len)])
		self.obs_mean = np.concatenate([self.state.obs_mean for i in range(self.pool_len)])
		self.obs_std = np.concatenate([self.state.obs_std for i in range(self.pool_len)])
		
		self.obs_mean = self.obs_mean*0
		self.obs_std = self.obs_std*0 + 1
		
		self.obs_dim = self.obs_mean.shape[0]
		self.act_dim = 12
		self.num_envs = 1
		
		
		motion_path = str(Path(__file__).parent) + "/motion/v1"
		self.reset_motion = np.load(motion_path+"/legs.npy")
		
		if self.debug:
			self.to_plot = [[] for i in range(100)]
		
		
		# --- setting up the adr ---
		#self.adr.add_param("act_rand_offset", 0, 0.003, 1000)
		#self.adr.add_param("act_rand_std", 0, 0.003, 1000)
		#self.adr.add_param("act_rand_offset", 0, 0, 1000)
		#self.adr.add_param("act_rand_std", 0, 0, 1000)
		
		
		self.adr.add_param("theta", 0, np.pi/100, np.pi)
		
		
		# --- training settings ---
		self.train_continuous = True
		self.train_speed = []
		self.train_rot_speed = []
		self.training_change_cmd = True
		self.only_forward = False
		self.has_rand_act_delta = False
		self.carthesian_act = True
		self.training_mode = 0
	
		
	
	def step(self, action):
		
		act = action.flatten()
		act_delta = self.state.act_offset # -np.ones(act.shape) * 0.1
		#act_delta = np.random.normal(size=act.shape) * self.adr.value("act_rand_std") + self.act_offset
		act_rand = act + act_delta
		legs_angle = self.kin.calc_joint_target (act_rand)
		#legs_angle = act_rand
		self.sim.step(act_rand, legs_angle)
		
		all_rew = [reward.step() for reward in self.rewards]
		rew = np.sum(all_rew)
		done = np.any([reward.done() for reward in self.rewards])
		
		#self.obs_pool = self.obs_pool[2:] + self.state.calc_obs() + [act]
		self.obs_pool = self.obs_pool[1:] + self.state.calc_obs()
		
		
		if self.debug:
			for reward, s in zip(all_rew, self.to_plot):
				s.append(reward)
		
		if self.test_adr:
			pos_speed_deviation = np.sum(np.square(self.state.target_speed - self.state.mean_planar_speed))
			rot_speed_deviation = np.square(self.state.target_rot_speed - self.state.mean_z_rot_speed)
			self.dev = pos_speed_deviation + rot_speed_deviation
			if pos_speed_deviation + rot_speed_deviation < np.square(0.2):
				self.frame_at_speed += 1
			adr_success = not done and self.frame_at_speed > 400*0.7
			self.adr.step(adr_success, not adr_success)
			"""
			self.cum_rew += rew
			adr_success = not done and self.cum_rew/self.state.frame > 0.7
			"""
		else:
			self.adr.step(False, False)
		
		if not self.debug:
			self.reset_cmd()
		
		
		return self.calc_obs(), [rew], [done]
		
	def reset(self):
		self.adr.reset()
		self.cum_rew = 0
		self.frame_at_speed = 0
		
		self.turn_rate = 1#self.adr.value("max_turning_rate")
		if not self.adr.is_test_param("max_turning_rate"):
			self.turn_rate *= np.random.random()
		if np.random.random() < .5:
			self.turn_rate *= -1
		if self.debug:
			self.turn_rate = 1
			#pass
			
		"""
		self.end_speed = 0
		self.end_rot = 0
		
		while self.end_speed*self.end_speed + self.end_rot*self.end_rot < 0.1*0.1:
			self.end_speed = np.random.random()
			self.end_rot = np.random.random()*2-1
		"""
		self.end_speed = np.random.random()
		self.end_rot = np.random.random()*2-1
		
		if self.end_speed*self.end_speed + self.end_rot*self.end_rot < 1000000 * 0.1*0.1:
			self.end_speed = 0
			self.end_rot = 0
		
		self.all_cmd = [self.choose_cmd(i) for i in range(4)]
		#self.secound_cmd = self.choose_cmd(100)
			
		
		self.kin.carthesian_act = self.carthesian_act
		if self.carthesian_act:
			#target_pose = np.asarray([0., 0., 0.3] * 4)
			#target_pose = np.asarray([0.5, 0.5, 0.3]*4)
			target_pose = np.asarray([0.5, 0.7, 0.3, 0.5, 0.3, 0.3]*2)
		"""
		else:
			target_pose = np.asarray([0.5, 0.8, 0.3]*4)
		"""
		
		
		des_v = 0#np.sqrt(np.sum(np.square(self.state.target_speed))) * np.random.random()
		des_clear = 0#0.05 * np.random.random()
		frame = int(np.random.random()*self.reset_motion.shape[0])
		#act = self.reset_motion[frame]
		act = target_pose# + np.random.normal(size=(12,))*0.1
		#legs_angle = self.kin.calc_joint_target (act)
		legs_angle = np.asarray(self.kin.calc_joint_target (act))
		self.sim.reset(des_v, des_clear, legs_angle)
		self.state.joint_fac = self.kin.standard_rot([1]*12)
		
		self.state.target_pose = target_pose
		self.state.mean_action = target_pose*1
		
		self.obs_pool = self.state.calc_obs()# + [act]
		
		# --- setting the target speed and rot ---
		if not self.debug:
			self.reset_cmd()
		
		for i in range(self.pool_len-1):
			self.sim.step(act, legs_angle)
			self.obs_pool += self.state.calc_obs()# + [act]
		
		
		#self.act_offset = (np.random.random(size=(12,))*2-1)*self.adr.value("act_rand_offset")
		if self.has_rand_act_delta:
			self.state.act_offset = (np.random.random(size=(12,))*2-1) * 0#.1
			self.state.joint_offset = (np.random.random(size=(12,))*2-1) * 0#.2
			self.state.loc_up_vect_offset = (np.random.random(size=(3,))*2-1) * 0.
		else:
			self.state.act_offset = np.ones((12,)) * 0#.1
			self.state.joint_offset = -np.ones((12,)) * 0#.05
			self.state.loc_up_vect_offset = (np.random.random(size=(3,))*2-1) * 0.
		
		return self.calc_obs()
	
	def choose_cmd (self, step=0):
		# self.training_mode (0:onlyforward, 1:smartforward, 2:allinplace)
		max_v_targ = 0.5
		if self.training_mode == 0:
			return (max_v_targ, 0, 0)
		elif self.training_mode == 1:
			r = np.random.random()
			"""
			if r < 1/6:
				return (0, 0, -1)
			elif r < 2/6:
				return (0, 0, 0)
			elif r < 3/6:
				return (0, 0, 1)
			elif r < 4/6:
				return (1, 0, -1)
			elif r < 5/6:
				return (1, 0, 0)
			elif r < 6/6:
				return (1, 0, 1)
			"""
			if r < 1/4:
				return (max_v_targ, 0, -np.random.random())
			elif r < 2/4:
				return (max_v_targ, 0, 0)
			elif r < 3/4:
				return (max_v_targ, 0, np.random.random())
			elif r < 4/4:
				return (0, 0, 0)
		elif self.training_mode == 2:
			if step == 0:
				return (max_v_targ, 0, 0)
			else:
				theta = self.adr.value("theta")
				if not step == 2: #not self.adr.is_test_param("theta"):
					theta *= np.random.random()
				if np.random.random() < 0.5:
					theta *= -1
				
				return (np.cos(theta)*max_v_targ, 0, np.sin(theta))
			
			"""
			v = np.random.normal(size=(3,))
			while np.sum(np.square(v)) < 0.0001:
				v = np.random.normal(size=(3,))
			v /= np.sqrt(np.sum(np.square(v)))
			v *= 0.7
			return (v[0], v[1], v[2])
			"""
	
	def reset_cmd (self):
		"""
		if self.state.frame < 100:
			targ_speed_x, targ_speed_y, targ_rot = self.first_cmd
		else:
			targ_speed_x, targ_speed_y, targ_rot = self.secound_cmd
		"""	
		i = min(self.state.frame//100, len(self.all_cmd)-1)
		targ_speed_x, targ_speed_y, targ_rot = self.all_cmd[i]
		self.state.target_speed = np.asarray([targ_speed_x, targ_speed_y])
		self.state.target_rot_speed = targ_rot
	
	def calc_obs (self):
		return [np.concatenate(self.obs_pool)]
	
	def close(self):
		self.adr.close()
	
	def set_epoch (self, ep):
		self.curr_ep = ep
		self.sim.set_epoch(ep)
		for rew in self.rewards:
			rew.set_epoch(ep)
	
	