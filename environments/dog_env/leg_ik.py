
import numpy as np

class Leg:
	def __init__ (self, inv_x=False, inv_y=False, inv_1=False, inv_2=False, inv_3=False):
		self.l1 = 0.196
		self.l2 = 0.180
		self.e = 0.02 + 0.098/2 # 0.069
		
		self.fac_1 = -1 if inv_1 else 1
		self.fac_2 = -1 if inv_2 else 1
		self.fac_3 = -1 if inv_3 else 1
		
		self.fac_x = -1 if inv_x else 1
		self.fac_y = -1 if inv_y else 1
		self.inv_y = inv_y
		
		self.create_range ()
	
	def motor_pos (self, action):
		foot_coord = self.calc_coord(action)
		motor_angle = self.calc_angle(foot_coord[:,0])
		return motor_angle
	
	# --- From raw action to cathesian coordinate ---
	def calc_coord (self, action):
		if self.inv_y:
			action[1] = 1-action[1]
		a = np.diag(action)
		u = (np.identity(3)-a) @ self.u_m + a @ self.u_M
		b = (np.identity(3)-a) @ self.b_m + a @ self.b_M
		x = np.linalg.solve (u, b)
		return x
	"""
	# --- From cathesian coordinate to rotation angle ---
	def calc_angle (self, coord): # coord : [x, y, z]
		a1 = np.arctan2(coord[1], -coord[2])[0]
		
		d2 = np.sum(np.square(coord))
		d = np.sqrt(d2)
		a3 = np.pi - np.arccos((self.l1*self.l1 + self.l2*self.l2 - d2)/(2*self.l1*self.l2))
		
		a_aux = np.arccos((self.l1*self.l1 - self.l2*self.l2 + d2)/(2*self.l1*d))
		a2 = np.arcsin(coord[0]/d)[0] - a_aux
		return [a1, -a2, -a3]
	"""
	
	# --- From cathesian coordinate to rotation angle ---
	def calc_angle (self, coord): # coord : [x, y, z]
		coord[0] *= self.fac_x
		#coord[1] *= self.fac_y
		
		lpx = np.sqrt(coord[1]**2 + coord[2]**2 - self.e**2)
		theta_1 = np.arctan2(lpx, self.e)
		theta_2 = np.arctan2(coord[1], -coord[2])
		a1 = np.pi/2 - theta_1 - theta_2
		
		d2 = lpx**2 + coord[0]**2
		d = np.sqrt(d2)
		a3 = np.pi - np.arccos((self.l1*self.l1 + self.l2*self.l2 - d2)/(2*self.l1*self.l2))
		
		a_aux = np.arccos((self.l1*self.l1 - self.l2*self.l2 + d2)/(2*self.l1*d))
		a2 = np.arcsin(coord[0]/d) - a_aux
		
		# inv calculations
		a1 = a1*self.fac_1
		a2 = a2*self.fac_2
		a3 = a3*self.fac_3
		
		return [a1, a2, -a3]
		
	# --- Creating the work volume ---
	def create_range (self):
		
		zM = -0.23
		ym = -0.069
		l = np.sqrt((self.l1+self.l2)**2+self.e**2) -0.01
		xm = 0.15
		zm = -0.32 #-np.sqrt(l*l-xm*xm-ym*ym)
		xM = 0.2 #np.sqrt(l*l-ym*ym-zM*zM)
		yl = np.sqrt(l**2 - zm**2 - xm**2)
		yh = np.sqrt(l**2 - zM**2 - xM**2)
		
		"""
		
		zM = -self.l1*2/3
		ym = 0.1
		l = self.l1+self.l2-0.01
		xm = 0.17
		zm = -np.sqrt(l*l-xm*xm-ym*ym)
		xM = np.sqrt(l*l-ym*ym-zM*zM)
		"""
		
		umx, bmx = self.face_from_point([-xm, yl, zm], [-xm, ym, zm], [-xM, yh, zM])
		uMx, bMx = self.face_from_point([xm, yl, zm], [xm, ym, zm], [xM, yh, zM])
		
		umy, bmy = self.face_from_point([-xm, ym, zm], [xm, ym, zm], [xM, ym, zM])
		uMy, bMy = self.face_from_point([-xm, yl, zm], [xm, yl, zm], [xM, yh, zM])
		
		#bmy += self.e*self.fac_y
		#bMy += self.e*self.fac_y
		
		umz, bmz = self.face_from_point([-xm, yl, zm], [xm, yl, zm], [xm, ym, zm])
		uMz, bMz = self.face_from_point([-xM, yh, zM], [xM, yh, zM], [xM, ym, zM])
		
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
		u /= np.sqrt(np.sum(np.square(u)))
		b = np.sum(p2*u)
		return u, b

	def standard_rot (self, rot):
		return [rot[0]*self.fac_1, rot[1]*self.fac_2, rot[2]*self.fac_3]