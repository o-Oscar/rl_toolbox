import numpy as np

class SimpleEnv:
	def __init__ (self):
		self.obs_dim = 2
		self.act_dim = 1
		self.num_envs = 1
		
		self.dt = 0.03
		
		self.curr_ep = 0
	
	def step (self, action):
		acc = ((action.flatten()[0]*2)-1)*10
		self.x += self.dt * self.v
		self.v += self.dt * acc
		
		rew = -(self.x*self.x + self.v*self.v/10)# + acc*acc)
		done = abs(self.x) > 1 or abs(self.v) > 1
		
		return self.calc_obs(), [rew], [done]
	
	def reset (self):
		self.x = np.minimum(np.maximum(np.random.normal()/3, -1), 1)
		self.v = np.minimum(np.maximum(np.random.normal()/3, -1), 1)
		
		return self.calc_obs()
	
	def calc_obs (self):
		return [[self.x, self.v]]
	
	
	def close(self):
		pass
	"""
	def set_epoch (self, ep):
		self.curr_ep = ep
	"""