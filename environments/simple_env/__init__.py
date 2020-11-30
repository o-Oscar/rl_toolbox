import numpy as np

class SimpleEnv:
	def __init__ (self):
		self.obs_dim = 1
		self.act_dim = 1
		self.num_envs = 1
		
		
		self.cur_frame = 0
	
	def step (self, action):
		acc = ((action.flatten()[0]*2)-1)
		
		rew = -np.square(acc-self.target)
		done = False # self.cur_frame > 3
		self.cur_frame += 1
		
		return self.calc_obs(), [rew], [done]
	
	def reset (self):
		self.target = int(np.random.random()*2)*2-1
		self.cur_frame = 0
		
		return self.calc_obs()
	
	def calc_obs (self):
		return [[self.target if self.cur_frame == 0 else 0]]
		#return [[self.target]]
	
	
	def close(self):
		pass
	"""
	def set_epoch (self, ep):
		self.curr_ep = ep
	"""