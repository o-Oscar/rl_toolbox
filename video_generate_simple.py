import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import time


from environments.simple_env import SimpleEnv
from models.actor import SimpleActor, MixtureOfExpert, LSTMActor
from ppo import PPO

if __name__ == "__main__":
	env = SimpleEnv()
	actor = LSTMActor(env)
	
	trainer = PPO(env, actor, init_log_std=-3)
	for i in range(10):
		all_s, all_a, all_r, all_neglog, all_masks = trainer.get_rollout(3)
		print(all_s)
		print(all_a)
		print(all_r)
		print(all_masks)