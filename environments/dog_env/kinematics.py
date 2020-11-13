import numpy as np
import time
import matplotlib.pyplot as plt

class Kinematics():
	def __init__(self):
		self.l1 = 0.167
		self.l2 = 0.18
		"""
		# une linge pour chaque vecteur normal à la face
		self.u_m = np.asarray([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
		self.u_M = np.asarray([[1, 0, -1], [0, 1, 0], [0, 0, 1]])
		
		# un nombre pour donner la "norme" d'un point de la face
		self.b_m = np.asarray([-0.4, -0.2, -0.3]).reshape((3,1))
		self.b_M = np.asarray([0.4, 0.2, -0.1]).reshape((3,1))
		"""
		self.create_range ()
		self.carthesian_act = True
	
	def calc_joint_target(self, action):
		
		flat_action = action.flatten()
		flat_action = np.maximum(np.minimum(flat_action, 1), 0)
		if self.carthesian_act:
			legs_angle = self.action_to_targ_angle_2 (flat_action) # _2
		else:
			legs_angle = self.action_to_targ_angle (flat_action) # _2
		#legs_angle = [0.0, 0.7, -2*0.7]*4
		return legs_angle
	
	# --- From raw action (int [0, 1]) to target angle for the legs ---
	def action_to_targ_angle (self, action):
		a_min = np.asarray([-0.7318217246298173, -0.8606353612620431, -2.492509298518673]*4)
		a_max = np.asarray([0.7318217246298173, 1.3606960000267483, -0.4816591423398249]*4)
		return action*(a_max-a_min)+a_min
	
	def action_to_targ_angle_2 (self, action):
		leg_actions = self.split_legs (action)
		legs_coord = list(map(self.calc_coord, leg_actions))
		#print(legs_coord[0])
		legs_angle = list(map(self.calc_angle, legs_coord))
		#print(legs_angle[0])
		return sum(legs_angle, [])
	
	# --- From one array of lenght 12 to one array of lenght 3 per leg ---
	def split_legs (self, action):
		to_return = []
		for i in range(4):
			to_return.append([action[i*3+j] for j in range(3)])
		return to_return
	
	# --- From raw action to cathesian coordinate ---
	def calc_coord (self, action):
		a = np.diag(action)
		u = (np.identity(3)-a) @ self.u_m + a @ self.u_M
		b = (np.identity(3)-a) @ self.b_m + a @ self.b_M
		x = np.linalg.solve (u, b)
		return x
		
	# --- From cathesian coordinate to rotation angle ---
	def calc_angle (self, coord):
		a1 = np.arctan2(coord[1], -coord[2])[0]
		
		d2 = np.sum(np.square(coord))
		d = np.sqrt(d2)
		a3 = np.pi - np.arccos((self.l1*self.l1 + self.l2*self.l2 - d2)/(2*self.l1*self.l2))
		
		a_aux = np.arccos((self.l1*self.l1 - self.l2*self.l2 + d2)/(2*self.l1*d))
		a2 = np.arcsin(coord[0]/d)[0] - a_aux
		return [a1, -a2, -a3]
	
	# --- Creating the work volume ---
	def create_range (self):
		zM = -self.l1*2/3
		ym = 0.1
		l = self.l1+self.l2-0.01
		xm = 0.17
		zm = -np.sqrt(l*l-xm*xm-ym*ym)
		xM = np.sqrt(l*l-ym*ym-zM*zM)
		
		umx, bmx = self.face_from_point([-xm, -ym, zm], [-xm, ym, zm], [-xM, ym, zM])
		uMx, bMx = self.face_from_point([xm, -ym, zm], [xm, ym, zm], [xM, ym, zM])
		
		umy, bmy = self.face_from_point([-xm, -ym, zm], [xm, -ym, zm], [xm, -ym, zM])
		uMy, bMy = self.face_from_point([-xm, ym, zm], [xm, ym, zm], [xm, ym, zM])
		
		umz, bmz = self.face_from_point([-xm, ym, zm], [xm, ym, zm], [xm, -ym, zm])
		uMz, bMz = self.face_from_point([-xm, ym, zM], [xm, ym, zM], [xm, -ym, zM])
		
		self.u_m = np.stack([umx, umy, umz])
		self.u_M = np.stack([uMx, uMy, uMz])
		
		self.b_m = np.asarray([bmx, bmy, bmz]).reshape((3,1))
		self.b_M = np.asarray([bMx, bMy, bMz]).reshape((3,1))
		
	def face_from_point (self, p1, p2, p3):
		p1 = np.asarray(p1).reshape((3,))
		p2 = np.asarray(p2).reshape((3,))
		p3 = np.asarray(p3).reshape((3,))
		v1 = p1-p2
		v2 = p3-p2
		u = np.cross(v1,v2)
		b = np.sum(p2*u)
		return u, b
	
if __name__ == "__main__":
	kin = Kinematics ()
	n = 2
	tests = [x/n for x in range(n+1)]
	joint_min = [0, 0, 0]
	joint_max = [0, 0, 0]
	for x in tests:
		for y in tests:
			for z in tests:
				legs_coord = kin.calc_coord([x, y, z])
				#print(legs_coord[0])
				legs_angle = kin.calc_angle(legs_coord)
				joint_min = np.minimum(joint_min, legs_angle)
				joint_max = np.minimum(joint_min, legs_angle)
	print(joint_min)
	print(joint_miax)