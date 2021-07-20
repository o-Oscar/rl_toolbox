import numpy as np
import os
import pickle

class CyclicLookup:
	def __init__ (self, data):
		self.data = data
		self._data_len = self.data.shape[0]

	def __len__(self):
		return self._data_len
	
	def phase_to_float_index (self, phase):
		return phase*self._data_len/(2*np.pi)
	def __getitem__(self, phase):
		f_index = self.phase_to_float_index(phase)
		m_index = int(f_index)%self._data_len
		M_index = (m_index+1)%self._data_len
		alpha = f_index%1.
		return (1-alpha)*self.data[m_index] + alpha*self.data[M_index]

class Reference:
	def __init__(self, qpos, qvel):
		self.qpos_getter = CyclicLookup(qpos)
		self.qvel_getter = CyclicLookup(qvel)
		assert len(self.qpos_getter) == len(self.qvel_getter), "qpos data and qvel data should be of the same length"
	
	def __len__ (self):
		return len(self.qpos_getter)
	def __getitem__(self, phase):
		return self.qpos_getter[phase], self.qvel_getter[phase]

class ReferenceBag:
	def __init__ (self):
		self.reference = {}
		self.loaded = False
		self.load()
		try:
			self.load()
		except:
			print("WARNING (ReferenceBag) : Was not able to load the reference")
	
	def load (self):
		with open(os.path.join("environments", "dog_env", "src", "motion", "walk.txt"), "rb") as f:
			saved = pickle.load(f)
			
			first_len = -1
			for name, (qpos, qvel) in saved["references"].items():
				self.reference[name] = Reference(qpos, qvel)
				if first_len < 0:
					first_len = len(self.reference[name])
				assert len(self.reference[name]) == first_len, "all references should have the same length"
			
			self.min_z = CyclicLookup(saved["min_z"])
		
		assert len(self.reference) > 0, "expected at least one reference"
		
		self.loaded = True
	
	def get_random_ref (self, phase):
		if not self.loaded:
			self.load()
		name = np.random.choice(list(self.reference.keys()))
		return self.get_ref(name, phase)
	
	def get_ref (self, name, phase):
		if not self.loaded:
			self.load()
		ref = self.reference[name]
		return ref[phase]