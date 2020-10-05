import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------------------ OBSERVATION CONFIG ------------------------------

"""
- Une observation c'est 
- Observation (obs) : n fois
	- Transition (trans) :
		- Etat (state) : 
			- Les 12 angles des moteurs
			- Le vecteur gravité dans le référentiel local
			- Le vecteur vitesse de rotation dans le référentiel local
			- La hauteur de la base par rapport au sol
			- La vitesse de la base dans le référentiel local
		- Action (action) :
			- Les 12 réels entre 0 et 1.
"""

JOINT_POS = 0
JOINT_POS_RAND = 1
LOCAL_UP = 2
LOCAL_UP_RAND = 3
ROT_VEL = 4
ROT_VEL_RAND = 5

POS_VEL_CMD = 6
ROT_VEL_CMD = 7

HEIGHT = 8
POS_VEL = 9
MEAN_POS_VEL = 10
MEAN_ROT_VEL = 11

state_vect = [JOINT_POS,
				JOINT_POS_RAND,
				LOCAL_UP,
				LOCAL_UP_RAND,
				ROT_VEL,
				ROT_VEL_RAND,
				POS_VEL_CMD,
				ROT_VEL_CMD,
				HEIGHT,
				POS_VEL,
				MEAN_POS_VEL,
				MEAN_ROT_VEL]

obs_transition_len = 3
