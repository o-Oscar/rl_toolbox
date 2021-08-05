import numpy as np
import time
from pathlib import Path

from .simulator import Simulator
from .kinematics import Kinematics
from .state import State
from .obs_gen import TeacherObsGenerator
from .reward import FullRewStandard
		

import gym
from gym import spaces


class DogEnv(gym.Env):
	def __init__(self, debug=False, render=False):
		super(DogEnv, self).__init__()
		
		self.debug = debug
		
		self.state = State()
		self.kin = Kinematics(self.state)
		self.sim = Simulator(self.state, self.debug)
		self.obs_gen = TeacherObsGenerator(self.state)
		
		self.reward = FullRewStandard(self.state)
		
		self.act_dim = 12
		self.obs_dim = self.obs_gen.obs_dim
		

		self.action_space = self.get_box_space (self.act_dim)
		# self.observation_space = spaces.Dict({"real_obs": self.get_box_space (self.obs_dim), "sim_obs": self.get_box_space (self.obs_dim)})
		self.observation_space = spaces.Dict({key:self.get_box_space(obs_dim) for key, obs_dim in self.obs_dim.items()})

	
	def get_box_space (self, shape):
		return spaces.Box(low=-np.ones(shape=shape).astype(np.float32), high=np.ones(shape=shape).astype(np.float32), dtype=np.float32)

	def step(self, action):
		action = np.maximum(np.minimum(action, 1), -1)
		act = action.flatten()
		
		targ_legs_angle = self.kin.calc_joint_target (act, self.state.phase)
		self.sim.step(targ_legs_angle, act)
		
		
		rew = self.reward.step()
		done = self.reward.done()
		
		return self.calc_obs(), rew, done, {}
		
	def reset(self, sim_args={}, render=True):
		
		self.state.sim_args = {
			"update_cmd": False, # out of date (probably)
			"ref_name": "in_place",
			"kin_use_reference" : True,
			"phase" : 0,
			# "kp" : 60,
			# "kd_fac": 0.12,
			# "gravity": [0, 0, -9.81],
		}
		self.state.sim_args.update(sim_args)
		
		if "action" in self.state.sim_args:
			self.state.sim_args["leg_pos"] = self.kin.calc_joint_target(self.state.sim_args["action"], 0)
		
		self.sim.reset(render)
		
		self.obs_gen.reset()
		
		obs = self.calc_obs()
		return obs
	
	def calc_obs (self):
		# return np.maximum(np.minimum(self.obs_gen.generate(), 1), -1)
		return self.obs_gen.generate()
	
	def close(self):
		self.sim.close()
	