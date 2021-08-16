from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

sim_out_size = 8

class DenseNetwork(nn.Module): # simple dense network
	def __init__(
		self,
		all_dims: List[int],
		last_layer_act: str = "none",
	):
		super(DenseNetwork, self).__init__()
		
		self.all_dims = all_dims

		all_layers = []
		for inp_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
			all_layers.append(nn.Linear(inp_dim, out_dim))
			all_layers.append(nn.ReLU())
		all_layers.pop()

		if last_layer_act == "none":
			pass
		elif last_layer_act == "relu":
			all_layers.append(nn.ReLU())
		elif last_layer_act == "tanh":
			all_layers.append(nn.Tanh())
		else:
			raise NameError("Type of layer activation {} not known.".format(last_layer_act))

		self.layers = nn.ModuleList(all_layers)

	def forward(self, inp: th.Tensor) -> th.Tensor:
		to_return = inp

		for layer in self.layers:
			to_return = layer(to_return)

		return to_return

class TeacherMonoExtractor (nn.Module): # preprocesses the observation only observed in observation before passing it to the main_network.
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(TeacherMonoExtractor, self).__init__()

		real_size = obs_space["real_obs"].low.shape[0]
		sim_size = obs_space["sim_obs"].low.shape[0]
		self.sim_network = DenseNetwork([sim_size, 64, sim_out_size], "relu")
		self.main_network = DenseNetwork([real_size + sim_out_size, 64, 64], "relu")

		# IMPORTANT:
		# Save output dimensions, used to create the distributions
		self.out_size = self.main_network.all_dims[-1]

	def forward(self, obs: th.Tensor) -> th.Tensor:
		sim_out = self.sim_network(obs["sim_obs"])
		return self.main_network(th.cat((obs["real_obs"], sim_out), dim=1)), sim_out

# vf_obs
class VfExtractor (nn.Module): # preprocesses the observation only observed in observation before passing it to the main_network.
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(VfExtractor, self).__init__()

		inn_size = obs_space["real_obs"].low.shape[0] + obs_space["vf_obs"].low.shape[0]
		self.main_network = DenseNetwork([inn_size, 64, 64], "relu")

		# IMPORTANT:
		# Save output dimensions, used to create the distributions
		self.out_size = self.main_network.all_dims[-1]

	def forward(self, obs: th.Tensor) -> th.Tensor:
		return self.main_network(th.cat((obs["real_obs"], obs["vf_obs"]), dim=1))


class TeacherMlpExtractor (nn.Module): # holds two extractors, one for the actor, one for the value function
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(TeacherMlpExtractor, self).__init__()

		self.pi_network = TeacherMonoExtractor(obs_space)
		self.vf_network = VfExtractor(obs_space)

		# IMPORTANT:
		# Save output dimensions, used to create the distributions
		self.latent_dim_pi = self.pi_network.out_size
		self.latent_dim_vf = self.vf_network.out_size

	def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
		return self.pi_network(obs)[0], self.vf_network(obs)


class MotorMonoExtractor (nn.Module): # preprocesses the observation only observed in observation before passing it to the main_network.
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(MotorMonoExtractor, self).__init__()

		motor_size = obs_space["motor_obs"].low.shape[0]
		self.main_network = DenseNetwork([motor_size, 64, 64], "relu")

		# IMPORTANT:
		# Save output dimensions, used to create the distributions
		self.out_size = self.main_network.all_dims[-1]

	def forward(self, obs: th.Tensor) -> th.Tensor:
		return self.main_network(obs["motor_obs"])

class MotorMlpExtractor (nn.Module): # holds two extractors, one for the actor, one for the value function
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(MotorMlpExtractor, self).__init__()

		self.pi_network = MotorMonoExtractor(obs_space)
		self.vf_network = VfExtractor(obs_space)

		# IMPORTANT:
		# Save output dimensions, used to create the distributions
		self.latent_dim_pi = self.pi_network.out_size
		self.latent_dim_vf = self.vf_network.out_size

	def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
		return self.pi_network(obs), self.vf_network(obs)









# -------------------------- Student  ! -------------------------- 

class CausalConv1d(th.nn.Conv1d):
	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size,
				 stride=1,
				 dilation=1,
				 groups=1,
				 bias=True):

		super(CausalConv1d, self).__init__(
			in_channels,
			out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=0,
			dilation=dilation,
			groups=groups,
			bias=bias)
		
		self.__padding = (kernel_size - 1) * dilation
		
	def forward(self, input):
		trans_input = th.transpose(input, 1, 2)
		trans_output = super(CausalConv1d, self).forward(th.nn.functional.pad(trans_input, (self.__padding, 0)))
		output = th.transpose(trans_output, 1, 2)

		return output
		
def conv_student_model (inn_size, out_size):
	# sum_l=1^L((k_l-1)*prod_i=1^l-1(s_i))
	# k_l : les kernel_size
	# s_i : les stride_size
	n_channels = 16
	return nn.Sequential(
			CausalConv1d(inn_size, n_channels, 5, stride=1, dilation=1),
			nn.ReLU(),
			# CausalConv1d(n_channels, n_channels, 5, stride=1, dilation=1),
			# nn.ReLU(),
			CausalConv1d(n_channels, n_channels, 5, stride=1, dilation=2),
			nn.ReLU(),
			# CausalConv1d(n_channels, n_channels, 5, stride=1, dilation=1),
			# nn.ReLU(),
			CausalConv1d(n_channels, out_size, 5, stride=1, dilation=4),
			nn.ReLU(),
			# CausalConv1d(n_channels, out_size, 5, stride=1, dilation=1),
			# nn.ReLU(),
		)


class StudentMonoExtractor (nn.Module):
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(StudentMonoExtractor, self).__init__()

		obs_size = obs_space.low.shape[0]
		# self.test_cnn_network = nn.Sequential(
		# 	nn.Linear(obs_size, 64),
		# 	nn.ReLU(),
		# 	CausalConv1d(64, 64, 3),
		# 	nn.ReLU(),
		# 	CausalConv1d(64, sim_out_size, 3),
		# 	nn.ReLU()
		# )
		self.cnn_network = conv_student_model(obs_size, sim_out_size)
		self.main_network = DenseNetwork([obs_size + sim_out_size, 64, 64], "relu")

		# IMPORTANT:
		# Save output dimensions, used to create the distributions
		self.out_size = self.main_network.all_dims[-1]

	def forward(self, obs: th.Tensor) -> th.Tensor:
		latent = self.cnn_network(obs)
		# test_latent = self.test_cnn_network(obs)
		# print(obs.shape)
		# print(latent.shape)
		# print(test_latent.shape)
		return self.main_network(th.cat((obs, latent), dim=2)), latent



class StudentModule (nn.Module):
	def __init__(
		self,
		obs_space: gym.spaces.Space,
	):
		super(StudentModule, self).__init__()

		self.mlp_extractor = StudentMonoExtractor(obs_space)
		self.action_net = nn.Linear(self.mlp_extractor.out_size, 12)


	def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
		extracted, latent = self.mlp_extractor(obs)
		return th.tanh(self.action_net(extracted)), latent