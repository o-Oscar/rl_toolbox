import math
"""
import gym
from gym import spaces, logger
from gym.utils import seeding
"""
import numpy as np
from scipy.integrate import odeint

clamp = lambda x, minx, maxx: max(min(maxx, x), minx)

# ------------------------------ OBSERVATION CONFIG ------------------------------

"""
- Une observation c'est 
	- Cart Position (max 0.3 m)
	- Cart Velocity (max 0.4 m.s-1)
	- Cos of Pole Angle (max 1)
	- Sin of Pole Angle (max 1)
	- Clamped Sin of Angle (between -0.1 and 0.1)
	- Rotation speed (max ? rad.s-2)
	- Clamped Rotation speed (max 0.5 rad.s-2)
	- Total normalised energy (between 0 and 1.5)
	- Clamped total normalised energy (between 0.95 and 1.05)
	- clamped (1-e*10)
	- mode (entrainement pour la stabilisation ou bien pour le swing up)
"""

class CartPoleEnv():
	def __init__(self):
		
		self.symetry = Symetry()
		self.blindfold = Blindfold()
		
		# --- obs stack ---
		self.obs_stack_len = 3
		self.obs_stack = []
		
		# --- env info ---
		self.obs_dim = 11 * self.obs_stack_len * 2
		self.act_dim = 1
		self.num_envs = 1
		self.adr = Adr()
		
		# --- setting up the adr ---
		self.adr.add_param("max_mrgsJ", 30.43, 0.2, 40)
		self.adr.add_param("min_mrgsJ", 30.43, -0.2, 20)
		
		self.adr.add_param("max_delay", 0, 0.0005, 0.02)
		a_pres = 2*np.pi/1200
		self.adr.add_param("max_a_noise", 0, a_pres/20, 2*a_pres)
		w_pres = a_pres/0.03
		self.adr.add_param("max_w_noise", 0, w_pres/20, 2*w_pres)
		
		nom_sol_fric = 0.023
		self.adr.add_param("min_solid_friction", nom_sol_fric, -nom_sol_fric/20, nom_sol_fric*1)
		self.adr.add_param("max_solid_friction", nom_sol_fric, nom_sol_fric/20, 3*nom_sol_fric)
		
		nom_flu_fric = 0.0062
		self.adr.add_param("min_fluid_friction", nom_flu_fric, -nom_flu_fric/20, nom_flu_fric*1)
		self.adr.add_param("max_fluid_friction", nom_flu_fric, nom_flu_fric/20, 3*nom_flu_fric)
		
		#self.adr.add_param("min_bench_len", 0.8, -0.005, 0.3)
		#self.adr.add_param("min_speed_range", 1, -0.005, 0.5)
		#self.adr.add_param("min_acc_range", 7500, -100, 0)
		
		
		
		self.gravity = 9.81
		self.nominal_dt = 0.03
		
		self.mrgsJ = 30.43 # valeur expérimentale
		self.mrsJ = self.mrgsJ/self.gravity
		
		self.x_total_max = 0.35
		self.v_max = 0.5 # 0.3
		self.acc_tau = 0.003 #0.003
		self.cpr = 2400
		
		self.max_v_cost = 0
		self.v_cost = 0
		
		# this is only used to rescale the observations
		self.max_usable_e = 1.9
		self.sin_clamp = 0.1
		self.min_e_clamp = 0.95
		self.max_e_clamp = 1.05
		
		blind = 2
		self.observation_low = np.array(([
			-self.x_total_max, -self.v_max,
			-1, -1, -self.sin_clamp,
			-np.sqrt(self.max_usable_e*4*self.mrgsJ), -1,
			0, self.min_e_clamp] + [0] + [0])*blind)
		self.observation_high = np.array(([
			self.x_total_max, self.v_max,
			1, 1, self.sin_clamp,
			np.sqrt(self.max_usable_e*4*self.mrgsJ), 1,
			self.max_usable_e, self.max_e_clamp] + [1] + [1])*blind)

		# 0 : swing up
		# 1 : stabilize
		self.mode = 0 # 1 if np.random.random() > c_mode else 0
		

	def dysdt (self, y, t, delta_v):
		a, w = y
		acc = delta_v/self.acc_tau*np.exp(-t/self.acc_tau)
		return [w, self.mrgsJ*np.sin(a) -self.mrsJ*acc*np.cos(a)- np.sign(w)*(abs(w)**2)*self.flu_fric - self.sol_fric*np.tanh(w/0.1)]
	
	
	def calc_next (self, x0, v0, a0, w0, v_targ, tau):
		delta_v = v_targ-v0
		x = x0 + tau*v_targ - delta_v*self.acc_tau*(1-np.exp(-tau/self.acc_tau))
		v = v_targ - delta_v*np.exp(-tau/self.acc_tau)
		a, w = odeint(self.dysdt, [a0, w0], [0, tau], args=(delta_v,))[1]
		return x, v, a, w
	
	def step(self, action, return_info=True):
		# SIMULATION
		x, v, a, w = self.state
		last_mes_a = np.round(a/(2*np.pi)*self.cpr)*(2*np.pi)/self.cpr
		
		# delay
		d_delay = 0 # np.random.normal()
		delay = self.delay + d_delay
		if delay > 0:
			x, v, a, w = self.calc_next(x, v, a, w, self.v_targ, delay)
	
		d_v_targ = 0 # np.random.uniform(-1, 1) * self.delta_v_action
		self.v_targ = max(min(action.flatten()[0]*2-1, 1), -1)*self.v_max + d_v_targ
		
		dt = self.nominal_dt-delay # + np.random.normal
		x, v, a, w = self.calc_next(x, v, a, w, self.v_targ, dt)
		
		self.state = (x,v,a,w)
		#self.n += 1
		
		da = np.random.normal()*self.noise_a
		mes_a = np.round((a + da)/(2*np.pi)*self.cpr)*(2*np.pi)/self.cpr
		mes_w = (mes_a-last_mes_a)/dt + np.random.normal()*self.noise_w
		
		mes_e = self.calc_e(x, v, mes_a, mes_w)
		e = self.calc_e(x, v, a, w)
		self.e = e
		#self.test = e
		
		# Calculating the reward
		reward = 0
		reward -= np.abs(e-1) * 0.2
		reward += (np.cos(a)/2-0.5) * 0.8
		"""
		reward -= np.square(e-1) * 0.4
		reward += np.cos(a) * 0.6
		reward -= np.square(v) * 0.3
		"""
		reward -= np.square(v) * 1e-2
		"""
		a_alpha = 0.1
		delta_action = new_action - last_action
		reward -= np.sqrt(np.square(delta_action/a_alpha) + 1)*a_alpha * 0.2
		"""
		
		# Calculation for the early stopping
		"""
		if np.cos(a) > 1-0.25:
			self.mode = 1
		if self.mode == 1:
			reward += 1
		"""
		
		self.done = self.done or e > self.max_usable_e
		self.done = self.done or abs(x) > self.x_total_max
		self.done = self.done or (self.mode == 1 and np.cos(a) < 1-0.5)
		
		if self.done:
			reward -= 1
		
		self.adr.step(reward, self.done)
		
		#self.stack_obs(x, v, cur_mes_a, mes_w, e)
		mes_x = x
		mes_v = v + (np.random.random()*2-1)*0.1
		self.stack_obs(x, v, a, w, e, mes_x, mes_v, mes_a, mes_w, mes_e)
		if return_info:
			return self.calc_obs(x, v, a, w, e), [reward], [self.done]
	
	def get_adr_param (self, name_min, name_max):
		to_return = np.random.random() * (self.adr.value(name_max) - self.adr.value(name_min)) + self.adr.value(name_min)
		if self.adr.is_test_param(name_max):
			to_return = self.adr.value(name_max)
		if self.adr.is_test_param(name_min):
			to_return = self.adr.value(name_min)
		return to_return
	
	def reset(self, zero=False, c_mode=0.5):
		
		self.adr.reset()
	
		self.mrgsJ = self.get_adr_param("min_mrgsJ", "max_mrgsJ")
		self.mrsJ = self.mrgsJ/self.gravity
		
		
		self.sol_fric = self.get_adr_param("min_solid_friction", "max_solid_friction")
		self.flu_fric = self.get_adr_param("min_fluid_friction", "max_fluid_friction")
		
		self.delay = self.adr.value("max_delay") * (1 if self.adr.is_test_param("max_delay") else np.random.random()) + 0.0075
		
		
		self.noise_a = self.adr.value("max_a_noise") * (1 if self.adr.is_test_param("max_a_noise") else np.random.random())
		self.noise_w = self.adr.value("max_w_noise") * (1 if self.adr.is_test_param("max_w_noise") else np.random.random())
		
		# actual reset 
		if self.mode == 0:
			r = np.random.random()
			if r < 0.3: # Setting the total energy
				e = 1
			else:
				e = (np.random.random()*1.5)
			
			if zero:
				e = 0
			
			epot = min(e, 1)*np.random.random() # Setting the potential energy
			a = np.arccos(2*epot-1) # Deduce the angle
			w = -np.sqrt(e*4*self.mrgsJ - 2*self.mrgsJ*(np.cos(a)+1)) # Deduce the rotation speed
			
			if np.random.random() < 1/2: # random direction
				a *= -1
			if np.random.random() < 1/2:
				w *= -1
			
			x = np.random.uniform(-self.x_total_max, self.x_total_max) # uniform for x and v
			v = np.random.uniform(-self.v_max, self.v_max)
			if zero:
				x = 0
				v = 0
			self.state = (x, v, a, w)
			self.v_targ = v
			self.last_a = a
		elif self.mode == 1:
			x = np.random.uniform(-self.x_total_max, self.x_total_max) # uniform for x and v
			v = np.random.uniform(-self.v_max, self.v_max)
			if zero:
				x = 0
				v = 0
			a = 0
			w = 0
			self.state = (x, v, a, w)
			self.v_targ = v
			self.last_a = a
			e = self.calc_e(x, v, a, w)
		
		self.done = False
		
		
		self.obs_stack = []
		for i in range(self.obs_stack_len):
			self.step(np.asarray([0.5]), return_info=False)
		
		return self.calc_obs(x, v, a, w, e)
	
	def calc_e (self, x, v, a, w):
		return (w**2)/(4*self.mrgsJ) + (np.cos(a)+1)/2
	
	def scale (self, x, box):
		return (x-box.low)/(box.high-box.low)
	
	def calc_true_obs (self, x, v, a, w, e):
		return [x, v, np.cos(a), np.sin(a), clamp(np.sin(a), -0.1, 0.1), w, clamp(w, -1, 1), e, clamp(e, 0.95, 1.05), clamp(1-e*10, 0, 1), self.mode]
	
	def calc_approx_obs (self, x, v, a, w):
		x += np.random.uniform(-1, 1) * self.delta_x
		v += np.random.uniform(-1, 1) * self.delta_v
		a += np.random.uniform(-1, 1) * self.delta_a + self.da
		w += np.random.uniform(-1, 1) * self.delta_w
		return self.calc_true_obs (x, v, a, w, self.calc_e(x, v, a, w))
		
	def stack_obs (self, x, v, a, w, e, mes_x, mes_v, mes_a, mes_w, mes_e):
		self.obs_stack.append((np.array(self.calc_true_obs(x, v, a, w, e) + self.calc_true_obs(mes_x, mes_v, mes_a, mes_w, mes_e))-self.observation_low)*2/(self.observation_high-self.observation_low) - 1)
		
	def calc_obs (self, x, v, a, w, e, true_obs=False):
		while len(self.obs_stack) > self.obs_stack_len:
			self.obs_stack = self.obs_stack[1:]
		if len(self.obs_stack) < self.obs_stack_len:
			print("ERROR : not enough stack", flush=True)
		return [np.concatenate(self.obs_stack)]

	def close(self):
		self.adr.close()

			

if __name__ == '__main__':
	print(calc_stack(1, 1, 0, 0.1, 1e-1))
else:
	from .symetry import Symetry
	from .blindfold import Blindfold
	from .adr import Adr