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
"""

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

	Actions:
		- target pos
		- cruse velocity
		- max acceleration
	"""
	
	def __init__(self, has_delay=True, is_random=False):
		
		self.symetry = Symetry()
		
		# --- obs stack ---
		self.obs_stack_len = 3
		self.obs_stack = []
		
		# --- env info ---
		self.obs_dim = 10 * self.obs_stack_len
		self.act_dim = 2
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
		self.adr.add_param("min_solid_friction", nom_sol_fric, -nom_sol_fric/20, 0)
		self.adr.add_param("max_solid_friction", nom_sol_fric, nom_sol_fric/20, 3*nom_sol_fric)
		
		nom_flu_fric = 0.0062
		self.adr.add_param("min_fluid_friction", nom_flu_fric, -nom_flu_fric/20, 0)
		self.adr.add_param("max_fluid_friction", nom_flu_fric, nom_flu_fric/20, 3*nom_flu_fric)
		
		self.adr.add_param("min_bench_len", 0.8, -0.005, 0.3)
		self.adr.add_param("min_speed_range", 1, -0.005, 0.5)
		#self.adr.add_param("min_acc_range", 7500, -100, 0)
		
		
		
		self.gravity = 9.81
		self.tau = 0.03
		
		self.da = 0
		self.mrgsJ = 30.43 # valeur expérimentale
		self.mrsJ = self.mrgsJ/self.gravity
		
		self.x_max = 0.05
		self.x_total_max = 0.35
		self.v_max = 1#0.3
		self.acc_tau = 0.003
		
		self.max_v_cost = 0
		self.v_cost = 0
		
		#used to add randomness to the obs and actions
		self.has_delay = has_delay
		rand_coef = 0
		if is_random:
			rand_coef = 1
		self.delta_v_action = self.v_max / 5000 * rand_coef
		
		self.delta_da = 2*np.pi/1200 * 2 * rand_coef
		self.delta_mrgsJ = 1e-2 * rand_coef
		
		self.delta_x = 1/5000 * rand_coef
		self.delta_v = self.v_max/5000 * rand_coef
		self.delta_a = 2*np.pi/1200 * rand_coef
		self.delta_w = 10/500 * rand_coef # 10/5000
		
		self.delta_delay = 0.003 * rand_coef
		
		#only for rendering purpuses
		self.length = 0.1
		
		#For early stopping (depreciated)
		self.mode = 0
		
		# this is only used to rescale the observations
		self.max_usable_e = 1.9
		self.sin_clamp = 0.1
		self.min_e_clamp = 0.95
		self.max_e_clamp = 1.05
		self.max_steps = 100
		"""
		observation_low = np.array([
			-self.x_total_max, -self.v_max,
			-1, -1, -self.sin_clamp,
			-np.sqrt(self.max_usable_e*4*self.mrgsJ), -1,
			0, self.min_e_clamp] + [0, 0])
		observation_high = np.array([
			self.x_total_max, self.v_max,
			1, 1, self.sin_clamp,
			np.sqrt(self.max_usable_e*4*self.mrgsJ), 1,
			self.max_usable_e, self.max_e_clamp] + [self.max_v_cost, self.max_steps])
		
		self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
		self.observation_space = spaces.Box(np.zeros(observation_low.shape), np.ones(observation_high.shape), dtype=np.float32)
		self.obs_rescale = spaces.Box(observation_low, observation_high, dtype=np.float32)
		"""
		self.observation_low = np.array([
			-self.x_total_max, -self.v_max,
			-1, -1, -self.sin_clamp,
			-np.sqrt(self.max_usable_e*4*self.mrgsJ), -1,
			0, self.min_e_clamp] + [0])
		self.observation_high = np.array([
			self.x_total_max, self.v_max,
			1, 1, self.sin_clamp,
			np.sqrt(self.max_usable_e*4*self.mrgsJ), 1,
			self.max_usable_e, self.max_e_clamp] + [1])
		"""
		self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
		self.observation_space = spaces.Box(np.zeros(observation_low.shape), np.ones(observation_high.shape), dtype=np.float32)
		self.obs_rescale = spaces.Box(observation_low, observation_high, dtype=np.float32)
		"""

		self.seed()
		self.viewer = None
		self.state = None
		self.n = None

		self.steps_beyond_done = None
		self.done = None

	def seed(self, seed=None):
		return [42]
	
	def dysdt (self, y, t, acc):
		a, w = y
		return [w, self.mrgsJ*np.sin(a) -self.mrsJ*acc*np.cos(a)- np.sign(w)*(abs(w)**2)*self.flu_fric - self.sol_fric*np.tanh(w/0.1)]
	
	def calc_next (self, x0, v0, a0, w0, x_cmd, v_cmd, acc_cmd, tau):
		if x0-x_cmd == 0 and v0 == 0:
			a, w = odeint(self.dysdt, [a0, w0], [0, tau], args=(0,))[1]
			return x0, v0, a, w
		
		stack = calc_stack(x0, v0, x_cmd, v_cmd, acc_cmd)
		
		curr_t = 0
		for x, v, acc, t in stack:
			if curr_t >= tau:
				pass
			elif curr_t + t >= tau:
				dt = tau - curr_t
				curr_t = tau
				x0 = x + v*dt + dt*dt*acc/2
				v0 = v + dt*acc
				a0, w0 = odeint(self.dysdt, [a0, w0], [0, dt], args=(acc, ))[1]
			elif t > 0:
				curr_t += t
				a0, w0 = odeint(self.dysdt, [a0, w0], [0, t], args=(acc, ))[1]
		
		if curr_t < tau:
			x0 = x_cmd
			v0 = 0
		
		return x0, v0, a0, w0
	
	def step(self, action, return_info=True):
		"""
		if self.done:
			x, v, a, w = self.state
			e = self.calc_e(x, v, a, w)
			return self.calc_obs(x, v, a, w, e), [0], [self.done]
		"""
		# SIMULATION
		x, v, a, w = self.state
		last_mes_a = np.round(a/(2*np.pi)*1200)*(2*np.pi)/1200
		
		# delay
		if self.delay > 0:
			x, v, a, w = self.calc_next(x, v, a, w, self.x_cmd, self.v_cmd, self.acc_cmd, self.delay)
	
		
		#self.v_targ = max(min(action.flatten()[0], 1), -1)*self.v_max + np.random.uniform(-1, 1) * self.delta_v_action
		self.x_cmd = max(min(action.flatten()[0]*2-1, 1), -1) * self.bench_len
		min_v = 1e-5
		self.v_cmd = min_v + max(min(action.flatten()[1], 1), 0) * (self.speed_range-min_v)
		self.acc_cmd = 7500 # max(min(action.flatten()[2], 1), -1) * self.acc_range
		
		
		x, v, a, w = self.calc_next(x, v, a, w, self.x_cmd, self.v_cmd, self.acc_cmd, self.tau-self.delay)
		
		self.state = (x,v,a,w)
		#self.n += 1
		
		
		cur_mes_a = np.round(a/(2*np.pi)*1200)*(2*np.pi)/1200 + np.random.normal()*self.delta_a
		mes_w = (cur_mes_a-last_mes_a)/self.tau + np.random.normal()*self.delta_w
		
		cur_mes_e = self.calc_e(x, v, cur_mes_a, mes_w)
		e = self.calc_e(x, v, a, w)
		#self.test = e
		
		# Calculating the reward
		reward = 0
		reward -= np.square(e-1) * 0.5
		reward -= (1-np.square(np.cos(a/2))) * 0.5
		"""
		a_alpha = 0.1
		delta_action = new_action - last_action
		reward -= np.sqrt(np.square(delta_action/a_alpha) + 1)*a_alpha * 0.2
		"""
		#u = last_action
		#alpha_u = 0.02
		#cost += alpha_u*alpha_u*(tf.cosh(u/alpha_u)-1)
		
		
		# Calculation for the early stopping
		
		self.done = self.done or e > self.max_usable_e
		self.done = self.done or abs(x) > self.x_total_max
		self.done = self.done or self.n >= self.max_steps
		
		if self.done:
			reward -= 1
		
		self.adr.step(reward, self.done)
		
		#self.stack_obs(x, v, cur_mes_a, mes_w, e)
		self.stack_obs(x, v, a, w, e)
		if return_info:
			return self.calc_obs(x, v, cur_mes_a, mes_w, e), [reward], [self.done]
	
	def fake_step (self, x, v, a, w, return_info=False):
		e = self.calc_e(x, v, a, w)
		self.stack_obs(x, v, a, w, e)
		if return_info:
			return self.calc_obs(x, v, a, w, e), [reward], [self.done]
	
	def get_adr_param (self, name_min, name_max):
		to_return = np.random.random() * (self.adr.value(name_max) - self.adr.value(name_min)) + self.adr.value(name_min)
		if self.adr.is_test_param(name_max):
			to_return = self.adr.value(name_max)
		if self.adr.is_test_param(name_min):
			to_return = self.adr.value(name_min)
		return to_return
	
	def reset(self, zero=False):
		self.adr.reset()
	
		self.v_cost = self.max_v_cost * np.random.uniform(0, 1)
		
		self.da = np.random.uniform(-1, 1) * self.delta_da
		#self.mrgsJ = 30.43 + np.random.uniform(-1, 1) * self.delta_mrgsJ # valeur expérimentale
		self.mrgsJ = self.get_adr_param("min_mrgsJ", "max_mrgsJ")
		self.mrsJ = self.mrgsJ/self.gravity
		
		
		self.sol_fric = self.get_adr_param("min_solid_friction", "max_solid_friction")
		self.flu_fric = self.get_adr_param("min_fluid_friction", "max_fluid_friction")
		
		self.delay = self.adr.value("max_delay") * (1 if self.adr.is_test_param("max_delay") else np.random.random()) + 0.0075
		
		
		self.delta_a = self.adr.value("max_a_noise") * (1 if self.adr.is_test_param("max_a_noise") else np.random.random())
		self.delta_w = self.adr.value("max_w_noise") * (1 if self.adr.is_test_param("max_w_noise") else np.random.random())
		
		self.bench_len = 0.8 - (0.8-self.adr.value("min_bench_len")) * (1 if self.adr.is_test_param("min_bench_len") else np.random.random())
		self.speed_range = 1 - (1-self.adr.value("min_speed_range")) * (1 if self.adr.is_test_param("min_speed_range") else np.random.random())
		#self.acc_range = 7500 - (7500-self.adr.value("min_acc_range")) * (1 if self.adr.is_test_param("min_acc_range") else np.random.random())
		
		
		
		r = np.random.random()
		if r < 0.1: # Setting the total energy
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
		self.n = 0
		
		self.x_cmd = x
		self.v_cmd = v
		self.acc_cmd = 7500
		
		
		self.steps_beyond_done = 0
		self.done = False
		
		self.obs_stack = []
		for i in range(self.obs_stack_len):
			self.step(np.asarray([0.5, 0.5]), return_info=False)
		
		return self.calc_obs(x, v, a, w, e)
	
	def calc_e (self, x, v, a, w):
		return (w**2)/(4*self.mrgsJ) + (np.cos(a)+1)/2
	
	def scale (self, x, box):
		return (x-box.low)/(box.high-box.low)
	
	def calc_true_obs (self, x, v, a, w, e):
		return [x, v, np.cos(a), np.sin(a), np.sin(clamp(a, -0.1, 0.1)), w, clamp(w, -1, 1), e, clamp(e, 0.95, 1.05), clamp(1-e*10, 0, 1)]
	
	def calc_approx_obs (self, x, v, a, w):
		x += np.random.uniform(-1, 1) * self.delta_x
		v += np.random.uniform(-1, 1) * self.delta_v
		a += np.random.uniform(-1, 1) * self.delta_a + self.da
		w += np.random.uniform(-1, 1) * self.delta_w
		return self.calc_true_obs (x, v, a, w, self.calc_e(x, v, a, w))
		
	def stack_obs (self, x, v, a, w, e):
		self.obs_stack.append((np.array(self.calc_true_obs(x, v, a, w, e))-self.observation_low)*2/(self.observation_high-self.observation_low) - 1)
	
	def calc_obs (self, x, v, a, w, e, true_obs=False):
		while len(self.obs_stack) > self.obs_stack_len:
			self.obs_stack = self.obs_stack[1:]
		if len(self.obs_stack) < self.obs_stack_len:
			print("ERROR : not enough stack", flush=True)
		return [np.concatenate(self.obs_stack)]

	def close(self):
		self.adr.close()
		if self.viewer:
			self.viewer.close()
			self.viewer = None


def calc_stack (x0, v0, x_cmd, v_cmd, acc_cmd):
	# trouver le domaine
	x0 -= x_cmd
	if v0 > 0:
		fac = 1 if x0 < -v0*v0/(2*acc_cmd) else -1
	else:
		fac = 1 if x0 < v0*v0/(2*acc_cmd) else -1
	if fac == -1:
		x0 = -x0
		v0 = -v0
	stack = []
	
	# gérer le cas ou v0 est plus grand que v_cmd
	if v0 > v_cmd:
		t0 = (v0-v_cmd)/acc_cmd
		stack.append([x0, v0, -acc_cmd, t0])
		x0 = x0 + t0*v0 - t0*t0*acc_cmd/2
		v0 = v_cmd
		
	# trouver l'intersection
	v_int = np.sqrt(v0*v0/2 - acc_cmd*x0)
	if v_int < v_cmd:
		t1 = (v_int-v0)/acc_cmd
		if t1 > 0:
			stack.append([x0, v0, acc_cmd, t1])
		x1 = x0 + v0*t1 + t1*t1*acc_cmd/2
		stack.append([x1, v_int, -acc_cmd, v_int/acc_cmd])
	else:
		t1 = 0
		if v0 < v_cmd:
			t1 = (v_cmd-v0)/acc_cmd	
			stack.append([x0, v0, acc_cmd, t1])
		x1 = x0 + (v_cmd-v0)*(v_cmd-v0)/(2*acc_cmd)
		x2 = -v_cmd*v_cmd/(2*acc_cmd)
		t2 = (x2-x1)/v_cmd
		stack.append([x1, v_cmd, 0, t2])
		t3 = v_cmd/acc_cmd
		stack.append([x2, v_cmd, -acc_cmd, t3])
	return [[fac*x+x_cmd, fac*v, fac*acc, t] for x, v, acc, t in stack]
		

if __name__ == '__main__':
	print(calc_stack(1, 1, 0, 0.1, 1e-1))
else:
	from .symetry import Symetry
	from .adr import Adr