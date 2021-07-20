import numpy as np

class CMD_Setter:
	def __init__ (self, all_state, debug):
		self.all_state = all_state
		self.debug = debug
		
		self.change_cmd_rate = 100
		
	def reset_cmd (self, init_cmd=np.asarray([0.4, 0, 0])):
		for state in self.all_state:
			state.target_rot_speed = np.asarray([0, 0, 0])
			state.target_speed = np.asarray(init_cmd)
		
	def update_cmd (self):
		"""
		if self.all_state[0].frame%self.change_cmd_rate == 0:
			theta = np.random.random()*np.pi*2
			target_speed = np.asarray([np.cos(theta), np.sin(theta)]) * 0.4
		"""
		target_speed = np.asarray([0.4, 0, 0])
		target_rot = np.asarray([0, 0, 0])
		for state in self.all_state:
			state.target_rot_speed = target_rot
			state.target_speed = target_speed