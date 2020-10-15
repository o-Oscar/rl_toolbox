
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path


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
		#self.adr.add_param("act_rand_offset", 0, 0.003, 1000)
		#self.adr.add_param("act_rand_std", 0, 0.003, 1000)
		self.adr.add_param("act_rand_offset", 0, 0, 1000)
		self.adr.add_param("act_rand_std", 0, 0, 1000)
		
		
		self.adr.add_param("max_turning_rate", 0, 0.01, 1)
		
		
		# --- training settings ---
		self.train_continuous = True
		self.train_speed = []
		self.train_rot_speed = []
		self.training_change_cmd = True
		self.only_forward = False
		
	
	def step(self, action):
		
		act = action.flatten()
		#act_delta = np.random.normal(size=act.shape) * self.adr.value("act_rand_std") + self.act_offset
		act_rand = act #+ act_delta
		legs_angle = self.kin.calc_joint_target (act_rand)
		self.sim.step(act_rand, legs_angle)
		
		all_rew = [reward.step() for reward in self.rewards]
		rew = np.sum(all_rew)
		done = np.any([reward.done() for reward in self.rewards])
		
		self.obs_pool = self.obs_pool[2:] + self.state.calc_obs() + [act]
		
		
		if self.debug:
			for reward, s in zip(all_rew, self.to_plot):
				s.append(reward)
		
		if self.test_adr:
			pos_speed_deviation = np.sum(np.square(self.state.target_speed - self.state.mean_planar_speed))
			rot_speed_deviation = np.square(self.state.target_rot_speed - self.state.mean_z_rot_speed)
			self.dev = pos_speed_deviation + rot_speed_deviation
			if pos_speed_deviation + rot_speed_deviation < np.square(0.2):
				self.frame_at_speed += 1
			adr_success = not done and self.frame_at_speed > 200
			self.adr.step(adr_success, not adr_success)
		else:
			self.adr.step(False, False)
		
		if not self.debug:
			self.reset_cmd()
		
		
		return self.calc_obs(), [rew], [done]
		
	def reset(self):
		self.adr.reset()
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
		
		if self.end_speed*self.end_speed + self.end_rot*self.end_rot < 0.1*0.1:
			self.end_speed = 0
			self.end_rot = 0
		
		
		des_v = 0#np.sqrt(np.sum(np.square(self.state.target_speed))) * np.random.random()
		des_clear = 0#0.05 * np.random.random()
		frame = int(np.random.random()*self.reset_motion.shape[0])
		act = self.reset_motion[frame]
		legs_angle = self.kin.action_to_targ_angle_2 (act)
		self.sim.reset(des_v, des_clear, legs_angle)
		
		self.obs_pool = self.state.calc_obs() + [act]
		
		# --- setting the target speed and rot ---
		if not self.debug:
			self.reset_cmd()
		
		for i in range(self.pool_len-1):
			self.sim.step(act, legs_angle)
			self.obs_pool += self.state.calc_obs() + [act]
		
		
		#self.act_offset = (np.random.random(size=(12,))*2-1)*self.adr.value("act_rand_offset")
		
		return self.calc_obs()
	
	def reset_cmd (self):
		"""
		if self.train_continuous:
			if np.random.random() < 0.1:
				targ_speed = 0
				targ_rot = 0
			else:
				targ_speed = 0.1 + np.random.random()
				targ_rot = (2*np.random.random()+0.1) * np.random.choice([-1, 1])
		else:
			targ_speed = np.random.choice(self.train_speed)
			targ_rot = np.random.choice(self.train_rot_speed)
		
		self.state.target_speed = np.asarray([1, 0]) * targ_speed
		self.state.target_rot_speed = targ_rot
		"""
		if self.only_forward:
			self.state.target_speed = np.asarray([1, 0]) * 1
			self.state.target_rot_speed = 0
		else:
			targ_speed = 1
			if self.state.frame < 100:
				targ_rot = 0
			elif self.state.frame < 200:
				targ_rot = -self.turn_rate
			elif self.state.frame < 300:
				targ_rot = self.turn_rate
			else :
				targ_speed = self.end_speed
				targ_rot = self.end_rot
				
			self.state.target_speed = np.asarray([1, 0]) * targ_speed
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
		
		if (ep+1)%100 == 0:
			self.adr.close()
	
	