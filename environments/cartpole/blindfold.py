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

stack_len = 3
are_seen = ([False]*10 + [False] + [True]*10 + [False]) * stack_len
width = np.sum([1 if is_seen else 0 for is_seen in are_seen])
height = len(are_seen)

obs_A = np.zeros((height, width))
j = 0
for i, is_seen in enumerate(are_seen):
	if is_seen:
		obs_A[i, j] = 1
		j += 1



class Blindfold:
	def action_blindfold (self, input_action):
		return input_action @ obs_A