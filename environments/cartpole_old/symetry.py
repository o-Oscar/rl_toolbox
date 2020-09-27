import numpy as np
import tensorflow as tf

"""
- Cart Position (max 0.3 m)
- Cart Velocity (max 0.4 m.s-1)
- Cos of Pole Angle (max 1)
- Sin of Pole Angle (max 1)
- Clamped Sin of Angle (between -0.1 and 0.1)
- Rotation speed (max ? rad.s-2)
- Clamped Rotation speed (max 0.5 rad.s-2)
- Total normalised energy (between 0 and 1.5)
- Clamped total normalised energy (between 0.95 and 1.05)
- Last action
- clamped (1-e*10)
"""

act_A = np.diag([-1, -1])
act_B = np.asarray([1, 1])

stack_len = 3
obs_A = np.diag([-1, -1, 1, -1, -1, -1, -1, 1, 1, -1] * stack_len)

class Symetry:
	def action_symetry (self, input_action):
		return input_action @ act_A + act_B

	def state_symetry (self, input_obs):
		return input_obs @ obs_A # + obs_B

	def loss (self, actor, input_obs, init_state, mask):
		diff = actor.model((input_obs, init_state))[0] - self.action_symetry(actor.model((self.state_symetry(input_obs), init_state))[0])
		return tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(diff), axis=-1), mask))/tf.reduce_mean(mask)
