
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

import pybullet as p

from .dogState import DogState
from .kinematics import Kinematics
from .simulator import Simulator
from .reward import NotDyingsRew, SpeedRew, TurningRew, PoseRew, JointAccRew, BaseAccRew, FootClearRew

from .symetry import Symetry
from .blindfold import Blindfold
from .adr import Adr

class DogEnv():
	def __init__(self, debug=False, render=False):
		self.symetry = Symetry()
		self.blindfold = Blindfold()
		
		self.debug = debug
		self.render = render
		
		self.adr = Adr()
		self.test_adr = False
		self.adr_rollout_len = 300
		
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
						FootClearRew(self.state)]
		
		self.curr_ep = 0
		
		
		# multiple obs and act in one obs
		self.obs_pool = []
		self.pool_len = 3 # config.obs_transition_len
		self.obs_mean = np.concatenate([self.state.obs_mean + [0.5 for j in range(12)] for i in range(self.pool_len)])
		self.obs_std = np.concatenate([self.state.obs_std + [1 for j in range(12)] for i in range(self.pool_len)])
		
		self.obs_dim = self.obs_mean.shape[0]
		self.act_dim = 12
		self.num_envs = 1
		
		
		motion_path = str(Path(__file__).parent) + "/motion/v1"
		self.reset_motion = np.load(motion_path+"/legs.npy")
		
		if self.debug:
			self.to_plot = [[] for i in range(100)]
		
		
		# --- setting up the adr ---
		self.adr.add_param("act_rand_offset", 0, 0.003, 1000)
		self.adr.add_param("act_rand_std", 0, 0.003, 1000)
		
		
	
	def step(self, action):
		
		act = action.flatten()
		act_delta = np.random.normal(size=act.shape) * self.adr.value("act_rand_std") + self.act_offset
		act_rand = act + act_delta
		legs_angle = self.kin.calc_joint_target (act_rand)
		self.sim.step(act_rand, legs_angle)
		
		all_rew = [reward.step() for reward in self.rewards]
		rew = np.sum(all_rew)
		done = np.any([reward.done() for reward in self.rewards])
		
		self.obs_pool = self.obs_pool[2:] + self.state.calc_obs() + [act]
		
		
		if self.debug:
			for reward, s in zip(all_rew, self.to_plot):
				s.append(reward)
		
		self.adr.step(rew, done)
		return self.calc_obs(), [rew], [done]
		
	def reset(self):
	
		des_v = 0#np.sqrt(np.sum(np.square(self.state.target_speed))) * np.random.random()
		des_clear = 0#0.05 * np.random.random()
		frame = int(np.random.random()*self.reset_motion.shape[0])
		act = self.reset_motion[frame]
		legs_angle = self.kin.action_to_targ_angle_2 (act)
		self.sim.reset(des_v, des_clear, legs_angle)
		
		self.obs_pool = self.state.calc_obs() + [act]
		
		for i in range(self.pool_len-1):
			self.sim.step(act, legs_angle)
			self.obs_pool += self.state.calc_obs() + [act]
		
		
		if 1: # training all main moves
			r = np.random.random()
			if r > 1/3:
				targ_speed = 1
			elif r > 2/3:
				targ_speed = 0.5
			else:
				targ_speed = 0
			self.state.target_speed = np.asarray([1, 0]) * targ_speed
				
			targ_rot = 0
			if np.random.random() > 1/2:
				if np.random.random() > 1/2:
					targ_rot = 1
				else:
					targ_rot = -1
			self.state.target_rot_speed = targ_rot
		if 0: # training only one move
			#targ_speed = 1
			targ_speed = 0
			self.state.target_speed = np.asarray([1, 0]) * targ_speed
			#targ_rot = 0
			targ_rot = -1
			self.state.target_rot_speed = targ_rot
			
		
		self.adr.reset()
		
		self.act_offset = (np.random.random(size=(12,))*2-1)*self.adr.value("act_rand_offset")
		
		return self.calc_obs()
	
	def calc_obs (self):
		return [np.concatenate(self.obs_pool)]
	
	def close(self):
		self.adr.close()
	
	def set_epoch (self, ep):
		self.curr_ep = ep
		self.sim.set_epoch(ep)
		for rew in self.rewards:
			rew.set_epoch(ep)
		
		if (ep+1)%100 == 0:
			self.adr.close()
	
	