from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import meshcat.transformations as tf
import numpy as np

class Visualizer:
	def __init__ (self, urdf_path, meshes_path):
		self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path, meshes_path)	
		self.viz = MeshcatVisualizer (self.model, self.collision_model, self.visual_model)
		try:
			self.viz.initViewer(open=False)
		except ImportError as err:
			print(err)
			exit(0)
		self.viz.loadViewerModel()
		
		# self.viz.viewer["/Cameras/default"].set_transform(np.asarray([1, 0, 0, 0,
		# 																0, 1, 0, 10,
		# 																0, 0, 1, 0,
		# 																0, 0, 0, 1]).reshape((4,4)).astype(np.float64))
		# self.viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0, 0, 10])
		# self.viz.viewer["/Cameras/default"].set_property("position", [0, 0, 10])
		# self.viz.viewer["/Cameras/default"].set_transform(tf.translation_matrix([0, 0, 10]))

		# print(list(vars(self.viz.viewer)))
		# help(self.viz.viewer.window.__dict__.keys())
		# exit()
		
	def update (self, q):
		self.viz.display(q)