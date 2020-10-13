import pybullet as p
import time
import pybullet_data
import numpy as np

def create_free_model ():
	with open("src_model.urdf") as f:
		model = f.read()
	
	m_scap = 0.3
	m_top = 0.3
	m_bot = 0.3
	
	model = model.replace("$all_foots", "")
	model = model.replace("$m_scap", str(m_scap))
	model = model.replace("$m_top", str(m_top))
	model = model.replace("$m_bot", str(m_bot))
	
	with open("robot.urdf", "w") as f:
		f.write(model)




if __name__ == "__main__":
	print(__file__)
	create_free_model()
	