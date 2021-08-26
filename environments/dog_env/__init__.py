import numpy as np
import time
from pathlib import Path

from .simulator import Simulator
from .kinematics import Kinematics
from .state import State
from .obs_gen import TeacherObsGenerator
from .reward import FullRewStandard
		
# git log --all --decorate --oneline --graph

import gym
from gym import spaces

class DogEnv(gym.Env):
	def __init__(self, motor_model=None, debug=False, render=False):
		super(DogEnv, self).__init__()
		
		self.debug = debug
		
		self.state = State()
		self.kin = Kinematics(self.state)
		self.sim = Simulator(self.state, self.debug)
		self.obs_gen = TeacherObsGenerator(self.state)
		
		self.reward = FullRewStandard(self.state)
		
		self.act_dim = 12
		self.obs_dim = self.obs_gen.obs_dim

		self.motor_model = motor_model

		self.action_space = self.get_box_space (self.act_dim)
		# self.observation_space = spaces.Dict({"real_obs": self.get_box_space (self.obs_dim), "sim_obs": self.get_box_space (self.obs_dim)})
		self.observation_space = spaces.Dict({key:self.get_box_space(obs_dim) for key, obs_dim in self.obs_dim.items()})
		

	
	def get_box_space (self, shape):
		return spaces.Box(low=-np.ones(shape=shape).astype(np.float32), high=np.ones(shape=shape).astype(np.float32), dtype=np.float32)

	def step(self, action):
		if self.motor_model is not None:
			motor_action = self.motor_model.predict(self.last_obs, deterministic=True)[0]

		action = np.maximum(np.minimum(action, 1), -1)
		act = action.flatten()
		
		targ_legs_angle = self.kin.calc_joint_target (act, self.state.phase)
		
		if self.motor_model is not None:
			self.sim.step(targ_legs_angle, act, motor_action = motor_action)
		else:
			self.sim.step(targ_legs_angle, act)
		
		
		rew = self.reward.step()
		done = self.reward.done()
		
		return self.calc_obs(), rew, done, {}
		
	def reset(self, sim_args={}, render=True):
		
		self.state.sim_args = {
			"update_cmd": False, # out of date (probably)
			"ref_name": "in_place",
			"kin_use_reference" : True,
			# "phase" : 0,
			# "kp" : 60,
			# "kd_fac": 0.12,
			"gravity": [0, 0, -9.81],
			"foot_f": [0.4]*4,
			"push_f": 250,
		}
		self.state.sim_args.update(sim_args)
		
		if "action" in self.state.sim_args:
			self.state.sim_args["leg_pos"] = self.kin.calc_joint_target(self.state.sim_args["action"], 0)
		
		self.sim.reset(render)
		
		self.obs_gen.reset()
		
		obs = self.calc_obs()
		return obs
	
	def calc_obs (self):
		self.last_obs = self.obs_gen.generate()
		# return np.maximum(np.minimum(self.obs_gen.generate(), 1), -1)
		return self.last_obs
	
	def close(self):
		self.sim.close()



from .reward import FollowRew
class DogEnv_follow(gym.Env):
	def __init__(self, debug=False, render=False):
		super(DogEnv_follow, self).__init__()
		
		self.debug = debug
		
		self.state = State()
		self.kin = Kinematics(self.state)
		self.sim = Simulator(self.state, self.debug)
		self.obs_gen = TeacherObsGenerator(self.state)

		import pickle
		import os
		with open(os.path.join("environments", "dog_env", "src", "proc_logs", "fall.txt"), "rb") as f:
			self.state.logged_fall = pickle.load(f)

		self.reward = FollowRew(self.state)
		
		self.act_dim = 12
		self.obs_dim = self.obs_gen.obs_dim
		

		self.action_space = self.get_box_space (self.act_dim)
		# self.observation_space = spaces.Dict({"real_obs": self.get_box_space (self.obs_dim), "sim_obs": self.get_box_space (self.obs_dim)})
		self.observation_space = spaces.Dict({key:self.get_box_space(obs_dim) for key, obs_dim in self.obs_dim.items()})

	
	def get_box_space (self, shape):
		return spaces.Box(low=-np.ones(shape=shape).astype(np.float32), high=np.ones(shape=shape).astype(np.float32), dtype=np.float32)

	def step(self, motor_action):
		action = self.state.logged_fall["actions"][self.state.frame]
		action = np.maximum(np.minimum(action, 1), -1)
		act = action.flatten()
		
		targ_legs_angle = self.kin.calc_joint_target (act, self.state.phase)
		self.sim.step(targ_legs_angle, act, motor_action)
		
		
		rew = self.reward.step()
		done = self.reward.done()
		
		return self.calc_obs(), rew, done, {}
		
	def reset(self, sim_args={}, render=True):
		
		self.state.sim_args = {
			"update_cmd": False, # out of date (probably)
			"ref_name": "in_place",
			"kin_use_reference" : True,
			"phase" : 0,
			"qpos": self.state.logged_fall["all_qpos"][0],
			# "kp" : 60,
			# "kd_fac": 0.12,
			# "gravity": [0, 0, -9.81],

			"kp":60,
			"kd_fac":0.12,
			# "base_state" : np.asarray([0, 0, 0.6, 0, 0, 0, 1]),
			# "reset_base" : True,
			# "update_phase": False,
			# "phase": np.pi,
			"foot_f": [0.3]*4,
			# "action" : np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
			"gravity": [0, 0, -9.81],
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
	