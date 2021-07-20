
import torch as th
from torch import nn
import numpy as np

def simple_student_model (inn_size):
	return nn.Sequential(
			nn.Linear(inn_size, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 12),
			nn.Tanh()
		)

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
		
def conv_student_model (inn_size):
	# sum_l=1^L((k_l-1)*prod_i=1^l-1(s_i))
	# k_l : les kernel_size
	# s_i : les stride_size
	return nn.Sequential(
			nn.Linear(inn_size, 64),
			nn.ReLU(),
			CausalConv1d(64, 64, 3),
			nn.ReLU(),
			CausalConv1d(64, 12, 3),
			nn.Tanh()
		)