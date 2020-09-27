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

# ------------------------------ TRAINING CONFIG ------------------------------

baseline = { # chien
	"use_symetry": True ,
	"use_blindfold": True ,
	
	"use_init_model": False,
	"init_model_path": "results/baseline_mix/models/{}_{}",
	
	"use_adr": True,
	"adr_sucess_threshold": 0.8, #-0.03, #-0.02,
	"adr_failure_threshold":0.75, # -0.025
	
	"ppo_train_step_nb": 12,
	"ppo_epoch_nb": 1000,
	"ppo_save_interval": 100,
	"adr_save_interval": 100,
	
	"rollout_len": 100,
	"rollout_nb": 8, #8,
	
	"lock_primitive": False,
	
	"default":0}
	
	
production = {
	"use_symetry": True ,
	
	"use_init_model": True,
	"init_model_path": "results/baseline/models/{}_{}",
	
	"use_adr": True,
	"adr_sucess_threshold": 0.8,
	"adr_failure_threshold": 0.75,
	
	"ppo_train_step_nb": 12,
	"ppo_epoch_nb": 1000,
	
	"default":0}
	
oneshot = {
	"use_symetry": True ,
	
	"use_init_model": False,
	"init_model_path": "results/baseline/models/{}_{}",
	
	"use_adr": True,
	"adr_sucess_threshold": 0.8,
	"adr_failure_threshold": 0.75,
	
	"ppo_train_step_nb": 12,
	"ppo_epoch_nb": 3000,
	
	"default":0}
	
test_simple = { # pendule inverse
	"use_symetry": True ,
	"use_blindfold": True ,
	
	"use_init_model": False,
	"init_model_path": "results/baseline_mix/models/{}_{}",
	
	"use_adr": True,
	"adr_sucess_threshold": 1000, #-0.03, #-0.02,
	"adr_failure_threshold":0.15, # -0.025
	
	"ppo_train_step_nb": 12,
	"ppo_epoch_nb": 300,
	"ppo_save_interval": 100,
	"adr_save_interval": 100,
	
	"rollout_len": 100,
	"rollout_nb": 8, #8,
	
	"lock_primitive": False,
	
	"default":0}
	
test_rnn = {
	"use_symetry": True,
	"use_blindfold": False ,
	
	"use_init_model": True,
	"init_model_path": "results/baseline/models/{}_{}",
	
	"use_adr": True,
	"adr_sucess_threshold": 100, #-0.12,
	"adr_failure_threshold":-0.18,
	
	"ppo_train_step_nb": 12,
	"ppo_epoch_nb": 50,
	"ppo_save_interval": 100,
	
	"rollout_len": 100,
	"rollout_nb": 100,
	
	"default":0}
	
training = baseline