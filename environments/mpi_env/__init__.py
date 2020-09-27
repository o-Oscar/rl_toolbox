from mpi4py import MPI
import numpy as np

RESET = 0
STEP = 1
CLOSE = 2
EPOCH = 3

class MpiEnv:

	def __init__ (self, root, env=None) :
		self.comm = MPI.COMM_WORLD
		
		if env is None:
			raise NameError("Can not create salve without env")
		if env.num_envs != 1:
			raise NameError("Sub-envs of a MpiEnv need to to contain only one env")
		
		self.rank = self.comm.Get_rank()
		self.root = root
		self.env = env
		
		self.obs_dim = self.env.obs_dim
		self.act_dim = self.env.act_dim
		self.num_envs = self.env.num_envs
		
		if hasattr(env, 'obs_mean'):
			self.obs_mean = env.obs_mean
		if hasattr(env, 'obs_std'):
			self.obs_std = env.obs_std
		if hasattr(env, 'act_a'):
			self.act_a = env.act_a
		if hasattr(env, 'act_b'):
			self.act_b = env.act_b
		
		personal_data = {"obs_dim": self.env.obs_dim, "act_dim": self.env.act_dim}
		all_data = self.comm.gather(personal_data, root=self.root)
		
		# --- gathering info in master rank ---
		if self.root == self.rank:
			for data in all_data:
				if self.obs_dim != data["obs_dim"] or self.act_dim != data["act_dim"]:
					raise NameError("Sub-envs of a MpiEnv need to have same obs_dim and act_dim")
			self.num_envs = len(all_data)
			print("MpiEnv successfully created with {} sub-env.".format(self.num_envs), flush=True)
		else:
			self.slave_loop()
		
		
	# --- master commands ---
	
	def reset (self):
		self.send_cmd(RESET)
		return self.joint_reset()
		# TODO : reset the env
	
	def step (self, all_action):
		self.send_cmd(STEP)
		return self.joint_step(all_action)
		"""
		action = np.empty((1, self.act_dim), dtype=np.float32)
		self.comm.Scatter(all_action, action, root=self.root)
		
		self.env.step(
		# TODO : step the env
		"""
		
	def close (self):
		self.send_cmd(CLOSE)
		self.joint_close()
		print("closing main env")
	
	def set_epoch (self, epoch):
		self.send_cmd(EPOCH)
		self.joint_set_epoch(epoch)
	
	def send_cmd (self, cmd):
		if self.root == self.rank:
			sendbuf = np.ones((self.num_envs,), dtype='i') * cmd
			recvbuf = np.empty((), dtype='i')
			self.comm.Scatter(sendbuf, recvbuf, root=self.root)
		else:
			raise NameError("Can't send cmd={} from slave MpiEnv".format(cmd))
	
	# --- slave main loop ---
	def slave_loop (self):
		end_requested = False
		while not end_requested:
			recvbuf = np.empty((), dtype='i')
			self.comm.Scatter(None, recvbuf, root=self.root)
			
			if recvbuf == RESET:
				self.joint_reset()
				
			if recvbuf == STEP:
				#self.slave_step()
				self.joint_step(None)
				
			if recvbuf == CLOSE:
				end_requested = True
				
			if recvbuf == EPOCH:
				self.joint_set_epoch(0)
		
		self.joint_close()
		print("closing slave {}".format(self.rank))
	
	# --- slave utility functions ---
	
	def joint_step (self, all_action):
		action = np.empty((1, self.act_dim), dtype=np.float32)
		self.comm.Scatter(all_action, action, root=self.root)
		
		obs, reward, done = self.env.step(action)
		obs = np.asarray(obs[0], dtype=np.float32)
		reward = np.asarray(reward[0], dtype=np.float32)
		done = np.asarray(done[0], dtype=np.bool)
		
		all_obs = None
		all_reward = None
		all_done = None
		if self.root == self.rank:
			all_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
			all_reward = np.empty((self.num_envs, ), dtype=np.float32)
			all_done = np.empty((self.num_envs, ), dtype=np.bool)
		
		self.comm.Gather(obs, all_obs, root=self.root)
		self.comm.Gather(reward, all_reward, root=self.root)
		self.comm.Gather(done, all_done, root=self.root)
		
		return all_obs, all_reward, all_done
	
	def joint_reset (self):
		obs = self.env.reset()
		obs = np.asarray(obs[0], dtype=np.float32)
		all_obs = None
		if self.root == self.rank:
			all_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
		self.comm.Gather(obs, all_obs, root=self.root)
		
		return all_obs
	
	def joint_set_epoch (self, epoch):
		sendbuf = np.ones((self.num_envs,), dtype='i') * epoch
		recvbuf = np.empty((), dtype='i')
		self.comm.Scatter(sendbuf, recvbuf, root=self.root)
		
		self.env.set_epoch (recvbuf)
	
	def joint_close (self):
		self.env.close()
	"""
	def joint_step (self):
		action = np.empty((1, self.act_dim), dtype=np.float32)
		self.comm.Scatter(None, recvbuf, root=self.root)
		
		obs, reward, done = self.env.step(action)
		slave_gather_info (self.env.step(action))
	"""
	