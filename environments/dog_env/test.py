"""
cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\raisimLib\\raisimUnity\\win32
RaiSimUnity.exe

python environments/dog_env_rai/test.py
"""

import os
import numpy as np


import raisimpy as raisim
import math
import time

# raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
world = raisim.World()
# robot = world.addArticulatedSystem("C://Users/oscbo/Documents/Travail/PSC/raisimLib/rsc/laikago/laikago.urdf")
robot = world.addArticulatedSystem("environments/dog_env_rai/src/dog_urdf/main_test.urdf")
#print("coucuo")

server = raisim.RaisimServer(world)
server.launchServer(8080)
server.focusOn(robot)

while (1):
	time.sleep(0.3)