import math
"""
import gym
from gym import spaces, logger
from gym.utils import seeding
"""
import numpy as np
from scipy.integrate import odeint

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
		
		self.obs_dim = 10
		self.act_dim = 1
		self.num_envs = 1
		
		self.gravity = 9.81
		self.tau = 0.03
		
		self.da = 0
		self.mrgsJ = 30.43 # valeur expérimentale
		self.mrsJ = self.mrgsJ/self.gravity
		
		self.x_max = 0.05
		self.x_total_max = 0.3
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
			self.max_usable_e, self.max_e_clamp] + [self.max_steps])
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
		"""
		if self.done:
			x, v, a, w = self.state
			e = self.calc_e(x, v, a, w)
			return self.calc_obs(x, v, a, w, e), [0], [self.done]
		"""
		# SIMULATION
		x, v, a, w = self.state
		last_mes_a = np.round(a/(2*np.pi)*1200)*(2*np.pi)/1200
		
		if self.has_delay:
			delay = 0.0075 + np.random.uniform(-1, 1) * self.delta_delay
			x, v, a, w = self.calc_next(x, v, a, w, self.v_targ, delay)
		else:
			delay = 0
		
		
		#self.v_targ = max(min(action.flatten()[0], 1), -1)*self.v_max + np.random.uniform(-1, 1) * self.delta_v_action
		self.v_targ = max(min(action.flatten()[0]*2-1, 1), -1)*self.v_max + np.random.uniform(-1, 1) * self.delta_v_action
		x, v, a, w = self.calc_next(x, v, a, w, self.v_targ, self.tau-delay)
		
		
		self.state = (x,v,a,w)
		#self.n += 1
		
		
		cur_mes_a = np.round(a/(2*np.pi)*1200)*(2*np.pi)/1200
		mes_w = (cur_mes_a-last_mes_a)/self.tau
		
		e = self.calc_e(x, v, cur_mes_a, mes_w)
		
		
		# Calculating the reward
		reward = 0
		reward -= np.square(e-1) * 0.5
		reward -= (1-np.square(np.cos(a/2))) * 0.5
		#alpha_u = 0.02
		#cost += tf.reduce_sum(alpha_u*alpha_u*tf.cosh(u/alpha_u)-1)
		
		"""
		reward = 0
		
		if abs(np.sin(a)) < 0.1 and np.cos(a) > 0:
			reward += 4 + 2-2*abs(self.v_targ)/self.v_max
		
		if e > 1.1:
			reward -= (e-1.1)/0.8
		if e < 0.9:
			reward -= (0.9-e)
		if abs(x) > self.x_max:
			reward *= (self.x_total_max-abs(x))/(self.x_total_max-self.x_max)
		
		reward -= abs(self.v_targ)*self.v_cost/self.v_max
		"""
		# Calculation for the early stopping
		"""
		if self.mode == 0 and abs(np.sin(a)) < 0.1 and np.cos(a) > 0:
			self.mode = 1
		if e > 1.1:
			self.mode = 0
		"""
		self.done = self.done or e > self.max_usable_e
		self.done = self.done or abs(x) > self.x_total_max
		self.done = self.done or self.n >= self.max_steps
		
		if self.done:
			reward -= 1
		
		"""
		# Throwing some debug if we step after the environement terminated
		if done:
			if self.steps_beyond_done == 1:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
		"""
		return self.calc_obs(x, v, cur_mes_a, mes_w, e), [reward], [self.done]

	def reset(self, zero=False):
		self.v_cost = self.max_v_cost * np.random.uniform(0, 1)
		
		self.da = np.random.uniform(-1, 1) * self.delta_da
		self.mrgsJ = 30.43 + np.random.uniform(-1, 1) * self.delta_mrgsJ # valeur expérimentale
		self.mrsJ = self.mrgsJ/self.gravity
		
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
		self.state = (x, v, a, w)
		self.v_targ = v
		self.last_a = a
		self.n = 0
		
		self.steps_beyond_done = 0
		self.done = False
		
		return self.calc_obs(x, v, a, w, e)
	
	def calc_e (self, x, v, a, w):
		return (w**2)/(4*self.mrgsJ) + (np.cos(a)+1)/2
	
	def scale (self, x, box):
		return (x-box.low)/(box.high-box.low)
	
	def calc_true_obs (self, x, v, a, w, e):
		return [x, v, np.cos(a), np.sin(a), np.sin(clamp(a, -0.1, 0.1)), w, clamp(w, -1, 1), e, clamp(e, 0.95, 1.05)]
	
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
		return [(np.array(self.calc_true_obs(x, v, a, w, e) + [self.n])-self.observation_low)/(self.observation_high-self.observation_low)]
		
		
	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_total_max*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			cart.set_color(.7,.2,.1)
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None: return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None



if __name__ == '__main__':
	env = CartPoleEnv()
	env.reset()
	env.render()
	while True:
		pass