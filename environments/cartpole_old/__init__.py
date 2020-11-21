import math
import numpy as np
from scipy.integrate import odeint

from .adr import Adr
from .symetry import Symetry

clamp = lambda x, minx, maxx: max(min(maxx, x), minx)

class CartPoleEnv():
	"""
	Description:
		A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The goal is to get the pendulum upright by controlling the cart velocity. Every constant are in SI.
		The pole has mass m, a moment of inertia J, a center of gravity at distance r of the cart.
		Thus, mrgsJ = m*r*g/J and mrsJ = m*r/J. It is quite interesting that only those two values govern the full kinematic of the problem.
		Friction is small and has been carefully tuned. It is comprised of a static force, proportional to sign(w) and a dynamic force, proportional to w**2.
		kinetic energy : (w**2)/(4*self.mrgsJ)
		potential energy : (np.cos(a)+1)/2
		
		
	Source:
		This environment was created by o_Oscar with the goal to transfert policies learned inside this model to the real world.

	Observation: 
		Type: Box(10)
		- Cart Position (max 0.3 m)
		- Cart Velocity (max 0.4 m.s-1)
		- Cos of Pole Angle (max 1)
		- Sin of Pole Angle (max 1)
		- Clamped Sin of Angle (between -0.1 and 0.1)
		- Rotation speed (max ? rad.s-2)
		- Clamped Rotation speed (max 0.5 rad.s-2)
		- Total normalised energy (between 0 and 1.5)
		- Clamped total normalised energy (between 0.95 and 1.05)
		- Time (from 0 to 3s)
		
	Actions:
		Type: Box(1)
		- Target velocity of the cart. Velocity is attained with exponential decay of carracteristic time acc_tau = 0.003s

	Reward:
		Reward is kinda wierd and carefully tuned. Please read the code.
	
	Starting State:
		Total normalised energy is chosen uniformly between 0 and 1.5 90% of the time and is set to equal 1 10% of the time.
		Proportion of potential and kinetic energy is chosen uniformly.
		Starting position and velocity is chosen uniformly in the full range of their allowed range.
		Note : setting the initial state to be as uniform as possible in the space of starting positions has a large impact in final performance.

	Episode Termination:
		Episode Termination is kinda wierd and carefully tuned. Please read the code.
		Note : Early stopping also has a huge impact on the training performance.
	"""
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 33
	}

	def __init__(self, has_delay=False, is_random=False):
		
		self.symetry = Symetry()
		
		self.stack_len = 1
		self.stack = []
		
		self.obs_dim = 9 * self.stack_len
		self.act_dim = 1
		self.num_envs = 1
		
		self.gravity = 9.81
		self.tau = 0.03
		
		self.mrgsJ = 30.43 # valeur expérimentale
		self.mrsJ = self.mrgsJ/self.gravity
		
		self.x_total_max = 0.3
		self.v_max = 1 # 0.3
		self.acc_tau = 0.003
		
		self.has_delay = True # has_delay		
		self.delay = 0.005 # 0.0075
		
		#only for rendering purpuses
		self.length = 0.1
		
		# this is only used to rescale the observations
		self.max_usable_e = 1.3
		self.sin_clamp = 0.1
		self.min_e_clamp = 0.95
		self.max_e_clamp = 1.05
		#self.max_steps = 100
		
		self.observation_low = np.array([
			-self.x_total_max, -self.v_max,
			-1, -1, -self.sin_clamp,
			-np.sqrt(self.max_usable_e*4*self.mrgsJ), -1,
			0, self.min_e_clamp])
		self.observation_high = np.array([
			self.x_total_max, self.v_max,
			1, 1, self.sin_clamp,
			np.sqrt(self.max_usable_e*4*self.mrgsJ), 1,
			self.max_usable_e, self.max_e_clamp])

		self.state = None
		self.n = None
		self.done = None
		
		self.adr = Adr()
		self.test_adr = False
		self.adr_rollout_len = 400
		self.n_good_frame = 0
		
		self.adr.add_param("x_min", 0, -self.x_total_max/100, -self.x_total_max)
		self.adr.add_param("x_max", 0, self.x_total_max/100, self.x_total_max)
		self.adr.add_param("v_min", 0, -self.v_max/100, -self.v_max)
		self.adr.add_param("v_max", 0, self.v_max/100, self.v_max)
		self.adr.add_param("e_max", 0, 1/100, 1)
		
		
	def dysdt (self, y, t, delta_v):
		a, w = y
		acc = delta_v/self.acc_tau*np.exp(-t/self.acc_tau)
		return [w, self.mrgsJ*np.sin(a) -self.mrsJ*acc*np.cos(a)- np.sign(w)*(abs(w)**2)*0.0062 - 0.023*np.tanh(w/0.1)]
	
	def calc_next (self, x0, v0, a0, w0, v_targ, tau):
		delta_v = v_targ-v0
		x = x0 + tau*v_targ - delta_v*self.acc_tau*(1-np.exp(-tau/self.acc_tau))
		v = v_targ - delta_v*np.exp(-tau/self.acc_tau)
		a, w = odeint(self.dysdt, [a0, w0], [0, tau], args=(delta_v,))[1]
		return x, v, a, w
	
	def step(self, action):
		# SIMULATION
		x, v, a, w = self.state
		last_mes_a = np.round(a/(2*np.pi)*1200)*(2*np.pi)/1200
		
		if self.has_delay:
			delay = self.delay # max 0.03
			x, v, a, w = self.calc_next(x, v, a, w, self.v_targ, delay)
		else:
			delay = 0
		
		
		#self.v_targ = max(min(action.flatten()[0], 1), -1)*self.v_max + np.random.uniform(-1, 1) * self.delta_v_action
		clamp_act = max(min(action.flatten()[0]*2-1, 1), -1)
		self.v_targ = clamp_act * self.v_max
		x, v, a, w = self.calc_next(x, v, a, w, self.v_targ, self.tau-delay)
		
		
		self.state = (x,v,a,w)
		self.n += 1
		
		
		cur_mes_a = np.round(a/(2*np.pi)*1200)*(2*np.pi)/1200
		mes_w = (cur_mes_a-last_mes_a)/self.tau
		
		mes_e = self.calc_e(x, v, cur_mes_a, mes_w)
		e = self.calc_e(x, v, a, w)
		
		# Calculating the reward
		reward = 0
		reward -= np.square(e-1) * 0.5
		reward -= (1-np.square(np.cos(a/2))) * 0.5
		reward += min(self.x_total_max-np.abs(x)-0.1, 0) * 5
		
		self.done = self.done or e > self.max_usable_e
		self.done = self.done or abs(x) > self.x_total_max
		
		if self.done:
			reward = -2
		
		if self.test_adr:
			if not self.done and np.cos(a) > 1-0.1:
				self.n_good_frame += 1
			adr_success = not self.done and self.n_good_frame > 150
			self.adr.step(adr_success, not adr_success)
		else:
			self.adr.step(False, False)
		
		return self.calc_obs(x, v, cur_mes_a, mes_w, e), [reward], [self.done]
		#return self.calc_obs(x, v, a, w, e), [reward], [self.done]

		
		
	def reset(self, zero=False):
	
		self.adr.reset()
		self.n_good_frame = 0
		
		self.mrgsJ = 30.43 # + np.random.uniform(-1, 1) * self.delta_mrgsJ # valeur expérimentale
		self.mrsJ = self.mrgsJ/self.gravity
		
		# --- setting the total energy ---
		"""
		r = np.random.random()
		if r < 0.1: 
			e = 1
		else:
			e = (np.random.random()*1.5)
		
		if zero:
			e = 0
		"""
		e = self.adr.value("e_max")*np.random.random()
		if self.adr.is_test_param("e_max"):
			e = self.adr.value("e_max")
			
		
		epot = min(e, 1)*np.random.random() # Setting the potential energy
		a = np.arccos(2*epot-1) # Deduce the angle
		w = -np.sqrt(e*4*self.mrgsJ - 2*self.mrgsJ*(np.cos(a)+1)) # Deduce the rotation speed
		
		if np.random.random() < 1/2: # random direction
			a *= -1
		if np.random.random() < 1/2:
			w *= -1
		
		x = np.random.uniform(-self.x_total_max, self.x_total_max) # uniform for x and v
		v = np.random.uniform(-self.v_max, self.v_max)
		
		# --- new initialisation ---
		x = np.random.uniform(self.adr.value("x_min"), self.adr.value("x_max"))
		v = np.random.uniform(self.adr.value("v_min"), self.adr.value("v_max"))
		
		if self.adr.is_test_param("x_min"):
			x = self.adr.value("x_min")
		if self.adr.is_test_param("x_max"):
			x = self.adr.value("x_max")
		if self.adr.is_test_param("v_min"):
			v = self.adr.value("v_min")
		if self.adr.is_test_param("v_max"):
			v = self.adr.value("v_max")
		
		self.state = (x, v, a, w)
		self.v_targ = v
		self.last_a = a
		self.n = 0
		
		self.done = False
		
		self.stack = []
		return self.calc_obs(x, v, a, w, e)
	
	def calc_e (self, x, v, a, w):
		return (w**2)/(4*self.mrgsJ) + (np.cos(a)+1)/2
	
	def scale (self, x, box):
		return (x-box.low)/(box.high-box.low)
	
	def calc_true_obs (self, x, v, a, w, e):
		return [x, v, np.cos(a), np.sin(a), clamp(np.sin(a), -0.1, 0.1), w, clamp(w, -1, 1), e, clamp(e, 0.95, 1.05)]
	
	def calc_approx_obs (self, x, v, a, w):
		x += np.random.uniform(-1, 1) * self.delta_x
		v += np.random.uniform(-1, 1) * self.delta_v
		a += np.random.uniform(-1, 1) * self.delta_a + self.da
		w += np.random.uniform(-1, 1) * self.delta_w
		return self.calc_true_obs (x, v, a, w, self.calc_e(x, v, a, w))
		
	def calc_obs (self, x, v, a, w, e, true_obs=False):
		#return np.array((x, v, np.cos(a), np.sin(a), np.sin(clamp(a, -0.1, 0.1)), w, clamp(w, -1, 1), e, clamp(e, 0.95, 1.05), self.n))
		#obs = np.array((x, v, np.cos(a), np.sin(a), np.sin(clamp(a, -0.1, 0.1)), w, clamp(w, -1, 1), e, clamp(e, 0.95, 1.05), self.n))
		"""
		if true_obs:
			return self.scale(self.calc_true_obs(x, v, a, w, e)*2 + [self.n], self.obs_rescale)
		else:
			#obs = self.calc_approx_obs(x, v, a, (a-self.last_a)/self.tau) + self.calc_true_obs(x, v, a, w, e) + [self.n]
			obs = self.calc_approx_obs(x, v, a, w) + self.calc_true_obs(x, v, a, w, e) + [self.n]
			return self.scale(np.array(obs), self.obs_rescale)
		"""
		if len(self.stack) == 0:
			for i in range(self.stack_len):
				self.stack.append((np.array(self.calc_true_obs(x, v, a, w, e))-self.observation_low)/(self.observation_high-self.observation_low))
		
		self.stack.append((np.array(self.calc_true_obs(x, v, a, w, e))-self.observation_low)/(self.observation_high-self.observation_low))
		self.stack = self.stack[-self.stack_len:]
		
		return [np.concatenate(self.stack, axis=-1)*2-1]
		
		
	def close(self):
		self.adr.close()



if __name__ == '__main__':
	env = CartPoleEnv()
	env.reset()
	env.render()
	while True:
		pass