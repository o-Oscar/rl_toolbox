
from .leg_ik import Leg
import numpy as np

class Kinematics():
	def __init__(self):
		self.all_legs = [Leg(inv_x=True, inv_y=False, inv_1=True, inv_2=False, inv_3=False),
					Leg(inv_x=True, inv_y=True, inv_1=False, inv_2=True, inv_3=False),
					Leg(inv_x=False, inv_y=False, inv_1=False, inv_2=True, inv_3=False),
					Leg(inv_x=False, inv_y=True, inv_1=True, inv_2=False, inv_3=False)]
			
		self.carthesian_act = True

	def calc_joint_target (self, action):
		return self.motor_pos(action)

	def motor_pos (self, raw_action):
		legs_actions = self.split_legs(raw_action)
		to_return = []
		for leg, leg_action in zip(self.all_legs, legs_actions):
			to_return += leg.motor_pos (leg_action)
		return np.asarray(to_return)

	def split_legs (self, action):
		to_return = []
		for i in range(4):
			to_return.append([action[i*3+j] for j in range(3)])
		return to_return
