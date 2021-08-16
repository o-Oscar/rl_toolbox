import numpy as np
import time
import os
import sys
import re

def is_int (s):
	try: 
		int(s)
		return True
	except ValueError:
		return False


class Config:
	def __init__(self, exp_name, models_names=["default"], debug=True):
		self.exp_name = exp_name
		self.path = os.path.join("results", self.exp_name)
		self.debug = debug
		"""
		if not os.path.isdir(self.path):
			if self.debug:
				print("Main experience folder not found. Creating it.")
			os.makedirs(self.path)
		"""

		self.models_names = models_names
		self.models_path = {name:os.path.join(self.path, name) for name in self.models_names}
		for name, model_path in self.models_path.items():
			if not os.path.isdir(model_path):
				if self.debug:
					print("Folder for model \"{}\" not found. Creating it.".format(name))
				os.makedirs(model_path)

		self.models_save_path = {name:os.path.join (self.models_path[name], "model_{epoch}") for name in self.models_names}
		self.models_best_path = {name:self.get_last_file (self.models_path[name], "model_{epoch}") for name in self.models_names}
		print(self.models_best_path)

	def get_last_file (self, path, save_name):
		pattern = save_name.format(epoch="([0-9]+)")

		prog = re.compile(pattern)
		all_id = []
		for filename in os.listdir(path):
			result = prog.search(filename)
			if result is not None and result.lastindex == 1 and is_int(result[1]):
				all_id.append(int(result[1]))
		if len(all_id) > 0:
			return os.path.join(path, save_name.format(epoch=max(all_id)))
		else:
			return None
