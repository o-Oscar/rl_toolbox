"""
cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\raisimLib\\raisimUnity\\win32
RaiSimUnity.exe

python environments\\dog_env_rai\\src\\dog_urdf\\create_urdf.py

"""

import os
import numpy as np
import matplotlib.pyplot as plt


import raisimpy as raisim
import math
import time

raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
world = raisim.World()
world.setGravity([0, 0, 0])
world.setTimeStep(0.001);

# dog = world.addArticulatedSystem("C://Users/oscbo/Documents/Travail/PSC/raisimLib/rsc/laikago/laikago.urdf")
dog = world.addArticulatedSystem("src/dog_urdf/main.urdf")
dog.setBasePos([0, 0, 1])
dog.setBaseOrientation([0, 0, 0, 1])
#pos, vel = dog.getState()
# getAngularVelocity
# getBaseOrientation
# getBodyIdx("trunk")
# getBodyNames
# getControlMode, setControlMode
# getFrameVelocity
# getState
# setBaseOrientation
# setBasePos
# setExternalForce
# setState
# getContacts

n_dof = dog.getDOF()
print(n_dof)
dog.setPdGains([100]*n_dof, [1]*n_dof)
dog.setPdTarget([0] * (n_dof+1-12) + [0.5]*12, [0]*n_dof)

ground = world.addGround()

SHOW = False
# launch raisim server
if SHOW:
	server = raisim.RaisimServer(world)
	server.launchServer(8080)

all_joint_pos = []
for i in range(2000):
	world.integrate()
	if i%20 == 0 and SHOW:
		time.sleep(0.02)
	pos, vel = dog.getState()
	all_joint_pos.append(pos[12:])
	#time.sleep(world.getTimeStep())

plt.plot(all_joint_pos)
plt.show()

if SHOW:
	server.killServer()